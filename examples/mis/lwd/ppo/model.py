# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Tuple

import dgl
import torch
from torch.distributions import Categorical
from torch.optim import Adam

from maro.rl.model import VNet, DiscretePolicyNet

from examples.mis.lwd.ppo.graph_net import GraphConvNet
from examples.mis.lwd.simulator import VertexState


VertexStateIndex = 0


def get_masks_idxs_subgraph_h(obs: torch.Tensor, graph: dgl.DGLGraph) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, dgl.DGLGraph, torch.Tensor,
]:
    """Extract masks and input feature information for the deferred vertexes.

    Args:
        obs (torch.Tensor): The observation tensor with shape (num_nodes, num_samples, feature_size).
        graph (dgl.DGLGraph): The input graph.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - undecided_node_mask, with shape (num_nodes, num_samples)
            - subgraph_mask, with shape (num_nodes)
            - subgraph_node_mask, with shape (num_nodes, num_samples)
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - flatten_node_idxs, with shape (num_nodes * num_samples)
            - flatten_subgraph_idxs, with shape (num_nodes)
            - flatten_subgraph_node_idxs, with shape (num_nodes * num_samples)
        dgl.DGLGraph:
        torch.Tensor: input tensor with shape (num_nodes, num_samples, 2)
    """
    # Mask tensor with shape (num_nodes, num_samples)
    undecided_node_mask = obs.select(2, VertexStateIndex).long() == VertexState.Deferred
    # Flatten index tensor with shape (num_nodes * num_samples)
    flatten_node_idxs = undecided_node_mask.reshape(-1).nonzero().squeeze(1)

    # Mask tensor with shape (num_nodes)
    subgraph_mask = undecided_node_mask.any(dim=1)
    # Flatten index tensor with shape (num_nodes)
    flatten_subgraph_idxs = subgraph_mask.nonzero().squeeze(1)

    # Mask tensor with shape (num_nodes, num_samples)
    subgraph_node_mask = undecided_node_mask.index_select(0, flatten_subgraph_idxs)
    # Flatten index tensor with shape (num_nodes * num_samples)
    flatten_subgraph_node_idxs = subgraph_node_mask.view(-1).nonzero().squeeze(1)

    # Extract a subgraph with only node in flatten_subgraph_idxs, batch_size -> 1
    subgraph = graph.subgraph(flatten_subgraph_idxs)

    # The observation of the deferred vertexes.
    h = obs.index_select(0, flatten_subgraph_idxs)

    num_nodes, num_samples = obs.size(0), obs.size(1)
    return subgraph_node_mask, flatten_node_idxs, flatten_subgraph_node_idxs, subgraph, h, num_nodes, num_samples


class GraphBasedPolicyNet(DiscretePolicyNet):
    def __init__(
        self,
        state_dim: int,
        action_num: int,
        hidden_dim: int,
        num_layers: int,
        init_lr: float,
    ) -> None:
        """A discrete policy net implemented with a graph as input.

        Args:
            state_dim (int): The dimension of the input state for this policy net.
            action_num (int): The number of pre-defined discrete actions, i.e., the size of the discrete action space.
            hidden_dim (int): The dimension of the hidden layers used in the GraphConvNet of the actor.
            num_layers (int): The number of layers of the GraphConvNet of the actor.
            init_lr (float): The initial learning rate of the optimizer.
        """
        super(GraphBasedPolicyNet, self).__init__(state_dim, action_num)

        self._actor = GraphConvNet(
            input_dim=state_dim,
            output_dim=action_num,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        with torch.no_grad():
            self._actor.layers[-1].bias[2].add_(3.0)

        self._optim = Adam(self._actor.parameters(), lr=init_lr)

    def _get_action_probs_impl(self, states: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def _get_actions_impl(self, states: torch.Tensor, exploring: bool, **kwargs) -> torch.Tensor:
        action, _ = self._get_actions_with_logps_impl(states, exploring, **kwargs)
        return action

    def _get_actions_with_probs_impl(
        self, states: torch.Tensor, exploring: bool, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _get_actions_with_logps_impl(
        self, states: torch.Tensor, exploring: bool, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert "graph" in kwargs, f"graph is required to given in kwargs"
        graph = kwargs["graph"]
        subg_mask, node_idxs, subg_node_idxs, subg, h, num_nodes, num_samples = get_masks_idxs_subgraph_h(states, graph)

        # Compute logits to get action, logits: shape (num_nodes * num_samples, 3)
        logits = self._actor(h, subg, mask=subg_mask).view(-1, self.action_num).index_select(0, subg_node_idxs)

        action = torch.zeros(num_nodes * num_samples, dtype=torch.long, device=self._device)
        action_log_probs = torch.zeros(num_nodes * num_samples, device=self._device)

        # NOTE: here we do not distinguish exploration mode and exploitation mode.
        # The main reason here for doing so is that the LwD modeling is learnt to better exploration,
        # the final result is chosen from the sampled trajectories.
        m = Categorical(logits=logits)
        action[node_idxs] = m.sample()
        action_log_probs[node_idxs] = m.log_prob(action.index_select(0, node_idxs))

        action = action.view(-1, num_samples)
        action_log_probs = action_log_probs.view(-1, num_samples)

        return action, action_log_probs

    def _get_states_actions_probs_impl(self, states: torch.Tensor, actions: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def _get_states_actions_logps_impl(self, states: torch.Tensor, actions: torch.Tensor, **kwargs) -> torch.Tensor:
        assert "graph" in kwargs, f"graph is required to given in kwargs"
        graph = kwargs["graph"]
        subg_mask, node_idxs, subg_node_idxs, subg, h, num_nodes, num_samples = get_masks_idxs_subgraph_h(states, graph)

        # compute logits to get action
        logits = self._actor(h, subg, mask=subg_mask).view(-1, self.action_num).index_select(0, subg_node_idxs)

        try:
            m = Categorical(logits=logits)
        except Exception:
            print(f"[GraphBasedPolicyNet] flatten_subgraph_node_idxs with shape {subg_node_idxs.shape}")
            print(f"[GraphBasedPolicyNet] logits with shape {logits.shape}")
            return None

        # compute log probability of actions per node
        actions = actions.reshape(-1)
        action_log_probs = torch.zeros(num_nodes * num_samples, device=self._device)
        action_log_probs[node_idxs] = m.log_prob(actions.index_select(0, node_idxs))
        action_log_probs = action_log_probs.view(-1, num_samples)

        return action_log_probs

    def get_actions(self, states: torch.Tensor, exploring: bool, **kwargs) -> torch.Tensor:
        actions = self._get_actions_impl(states, exploring, **kwargs)
        return actions

    def get_states_actions_logps(self, states: torch.Tensor, actions: torch.Tensor, **kwargs) -> torch.Tensor:
        logps = self._get_states_actions_logps_impl(states, actions, **kwargs)
        return logps


class GraphBasedVNet(VNet):
    def __init__(self, state_dim: int, hidden_dim: int, num_layers: int, init_lr: float, norm_base: float) -> None:
        """A value net implemented with a graph as input.

        Args:
            state_dim (int): The dimension of the input state for this value network.
            hidden_dim (int): The dimension of the hidden layers used in the GraphConvNet of the critic.
            num_layers (int): The number of layers of the GraphConvNet of the critic.
            init_lr (float): The initial learning rate of the optimizer.
            norm_base (float): The normalization base for the predicted value. The critic network will predict the value
                of each node, and the returned v value is defined as `Sum(predicted_node_values) / normalization_base`.
        """
        super(GraphBasedVNet, self).__init__(state_dim)
        self._critic = GraphConvNet(
            input_dim=state_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        self._optim = Adam(self._critic.parameters(), lr=init_lr)
        self._normalization_base = norm_base

    def _get_v_values(self, states: torch.Tensor, **kwargs) -> torch.Tensor:
        assert "graph" in kwargs, f"graph is required to given in kwargs"
        graph = kwargs["graph"]
        subg_mask, node_idxs, subg_node_idxs, subg, h, num_nodes, num_samples = get_masks_idxs_subgraph_h(states, graph)

        values = self._critic(h, subg, mask=subg_mask).view(-1).index_select(0, subg_node_idxs)
        # Init node value prediction, shape (num_nodes * num_samples)
        node_value_preds = torch.zeros(num_nodes * num_samples, device=self._device)
        node_value_preds[node_idxs] = values

        graph.ndata["h"] = node_value_preds.view(-1, num_samples)
        value_pred = dgl.sum_nodes(graph, "h") / self._normalization_base
        graph.ndata.pop("h")

        return value_pred

    def v_values(self, states: torch.Tensor, **kwargs) -> torch.Tensor:
        v = self._get_v_values(states, **kwargs)
        return v
