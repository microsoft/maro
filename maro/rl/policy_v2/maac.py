# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from collections import defaultdict
from typing import Callable, Iterable, List, Tuple

import numpy as np
import torch

from maro.rl.modeling_v2 import DiscretePolicyGradientNetwork
from maro.rl.modeling_v2.critic_model import MultiDiscreteQCriticNetwork
from maro.rl.policy_v2 import AbsRLPolicy
from maro.rl.policy_v2.buffer import MultiBuffer
from maro.rl.policy_v2.policy_interfaces import MultiDiscreteActionMixin
from maro.rl.utils import average_grads


class MultiDiscreteActorCritic(MultiDiscreteActionMixin, AbsRLPolicy):
    """
    References:
        MADDPG paper: https://arxiv.org/pdf/1706.02275.pdf

    Args:
        name (str): Unique identifier for the policy.
        global_state_dim (int): State dim of the shared part of state.
        agent_nets (List[DiscretePolicyGradientNetwork]): Networks for all sub-agents.
        critic_net (MultiDiscreteQCriticNetwork): Critic's network.
        reward_discount (float): Reward decay as defined in standard RL terminology.
        grad_iters (int): Number of gradient steps for each batch or set of batches. Defaults to 1.
        min_logp (float): Lower bound for clamping logP values during learning. This is to prevent logP from becoming
            very large in magnitude and causing stability issues. Defaults to None, which means no lower bound.
        critic_loss_cls: A string indicating a loss class provided by torch.nn or a custom loss class for computing
            the critic loss. If it is a string, it must be a key in ``TORCH_LOSS``. Defaults to "mse".
        critic_loss_coef (float): Coefficient of critic loss.
        clip_ratio (float): Clip ratio in the PPO algorithm (https://arxiv.org/pdf/1707.06347.pdf). Defaults to None,
            in which case the actor loss is calculated using the usual policy gradient theorem.
        lam (float): Lambda value for generalized advantage estimation (TD-Lambda). Defaults to 0.9.
        max_trajectory_len (int): Maximum trajectory length that can be held by the buffer (for each agent that uses
            this policy). Defaults to 10000.
        get_loss_on_rollout (bool): If True, ``get_rollout_info`` will return the loss information (including gradients)
            for the trajectories stored in the buffers. The loss information, along with that from other roll-out
            instances, can be passed directly to ``update``. Otherwise, it will simply process the trajectories into a
            single data batch that can be passed directly to ``learn``. Defaults to False.
        device (str): Identifier for the torch device. The ``ac_net`` will be moved to the specified device. If it is
            None, the device will be set to "cpu" if cuda is unavailable and "cuda" otherwise. Defaults to None.
    """
    def __init__(
        self,
        name: str,
        global_state_dim: int,
        agent_nets: List[DiscretePolicyGradientNetwork],
        critic_net: MultiDiscreteQCriticNetwork,
        reward_discount: float,
        grad_iters: int = 1,
        min_logp: float = None,
        critic_loss_cls: Callable = None,
        critic_loss_coef: float = 1.0,
        clip_ratio: float = None,
        lam: float = 0.9,
        max_trajectory_len: int = 10000,
        get_loss_on_rollout: bool = False,
        device: str = None
    ) -> None:
        super(MultiDiscreteActorCritic, self).__init__(name=name, device=device)

        self._critic_net = critic_net
        self._total_state_dim = self._critic_net.state_dim
        self._global_state_dim = global_state_dim

        self._agent_nets = agent_nets
        self._num_sub_agents = len(self._agent_nets)
        self._local_state_dims = [net.state_dim - self._global_state_dim for net in self._agent_nets]
        assert all(dim >= 0 for dim in self._local_state_dims)
        assert self._total_state_dim == sum(self._local_state_dims) + self._global_state_dim

        self._reward_discount = reward_discount
        self._grad_iters = grad_iters
        self._min_logp = min_logp
        self._critic_loss_func = critic_loss_cls() if critic_loss_cls is not None else torch.nn.MSELoss()
        self._critic_loss_coef = critic_loss_coef
        self._clip_ratio = clip_ratio
        self._lam = lam
        self._max_trajectory_len = max_trajectory_len
        self._get_loss_on_rollout = get_loss_on_rollout

        self._buffer = defaultdict(lambda: MultiBuffer(agent_num=self._num_sub_agents, size=self._max_trajectory_len))

    def _get_action_nums(self) -> List[int]:
        return [net.action_num for net in self._agent_nets]

    def _get_state_dim(self) -> int:
        return self._critic_net.state_dim

    def _call_impl(self, states: np.ndarray) -> Iterable:
        actions, logps, values = self.get_actions_with_logps_and_values(states)
        return [
            {
                "action": action,  # [num_sub_agent]
                "logp": logp,  # [num_sub_agent]
                "value": value  # Scalar
            } for action, logp, value in zip(actions, logps, values)
        ]

    def _get_state_list(self, input_states: np.ndarray) -> List[torch.Tensor]:
        """Get observable states for all sub-agents.

        Args:
            input_states (np.ndarray): global state with shape [batch_size, total_state_dim]

        Returns:
            A list of torch.Tensor.

        """
        state_list = []
        global_state = input_states[:, -self._global_state_dim]  # [batch_size, global_state_dim]
        offset = 0
        for local_state_dim in self._local_state_dims:
            local_state = input_states[:, offset:offset + local_state_dim]  # [batch_size, local_state_dim]
            offset += local_state_dim

            complete_state = np.concatenate([local_state, global_state], axis=1)  # [batch_size, complete_state_dim]
            complete_state = self.ndarray_to_tensor(complete_state)
            if len(complete_state.shape) == 1:
                complete_state = complete_state.unsqueeze(dim=0)
            state_list.append(complete_state)
        return state_list

    def get_actions_with_logps_and_values(self, input_states: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """

        Args:
            input_states (np.ndarray): global state with shape [batch_size, total_state_dim]

        Returns:
            actions: [batch_size, num_sub_agent]
            logps: [batch_size, num_sub_agent]
            values: [batch_size]
        """
        for net in self._agent_nets:
            net.eval()

        state_list = self._get_state_list(input_states)
        with torch.no_grad():
            actions = []
            logps = []
            for net, state in zip(self._agent_nets, state_list):  # iterate `num_sub_agent` times
                action, logp = net.get_actions_and_logps(state, self._exploring)  # [batch_size], [batch_size]
                actions.append(action)
                logps.append(logp)
            values = self._get_values_by_states_and_actions(self.ndarray_to_tensor(input_states), actions)

        actions = np.stack([action.cpu().numpy() for action in actions], axis=1)  # [batch_size, num_sub_agent]
        logps = np.stack([logp.cpu().numpy() for logp in logps], axis=1)  # [batch_size, num_sub_agent]
        values = values.cpu().numpy()  # [batch_size]

        return actions, logps, values

    def _get_values_by_states_and_actions(self, states: torch.Tensor, actions: List[torch.Tensor]) -> torch.Tensor:
        """
        states: [batch_size, state_dim]
        actions: List of torch.Tensor with shape [batch_size]

        Returns:
            [batch_size]
        """
        action_tensor = torch.stack(actions).T  # [batch_size, sub_agent_num]
        return self._critic_net.q_critic(states, action_tensor)

    def record(
        self, key: str, state: np.ndarray, action: dict, reward: float,
        next_state: np.ndarray, terminal: bool
    ) -> None:
        self._buffer[key].put(state, action, reward, terminal)

    def get_rollout_info(self) -> dict:
        if self._get_loss_on_rollout:
            return self.get_batch_loss(self._get_batch(), explicit_grad=True)
        else:
            return self._get_batch()

    def _get_batch(self) -> dict:
        batch = defaultdict(list)
        for buf in self._buffer.values():
            trajectory = buf.get()
            batch["states"].append(trajectory["states"][:-1])
            batch["actions"].append(trajectory["actions"][:-1])
            batch["next_states"].append(trajectory["next_states"][1:])
            batch["next_actions"].append(trajectory["next_actions"][1:])
            batch["rewards"].append(trajectory["rewards"][:-1])
            batch["terminals"].append(trajectory["terminals"][:-1])
        return {key: np.concatenate(vals) for key, vals in batch.items()}  # batch_size = sum(buffer_length)

    def get_batch_loss(self, batch: dict, explicit_grad: bool = False) -> dict:
        for i, net in enumerate(self._agent_nets):
            net.train()
        self._critic_net.train()

        states_ndarray = batch["states"]
        states = self.ndarray_to_tensor(batch["states"])  # [batch_size, total_state_dim]
        actions = [self.ndarray_to_tensor(elem).long() for elem in batch["actions"]]
        next_states = self.ndarray_to_tensor(batch["next_states"])  # [batch_size, total_state_dim]
        next_actions = [self.ndarray_to_tensor(elem).long() for elem in batch["next_actions"]]

        rewards = self.ndarray_to_tensor(batch["rewards"])  # [batch_size]
        terminals = self.ndarray_to_tensor(batch["terminals"]).float()  # [batch_size]

        # critic loss
        with torch.no_grad():
            next_q_values = self._get_values_by_states_and_actions(next_states, next_actions)
        target_q_values = (rewards + self._reward_discount * (1 - terminals) * next_q_values).detach()  # [batch_size]
        q_values = self._get_values_by_states_and_actions(states, actions)  # [batch_size]
        critic_loss = self._critic_loss_func(q_values, target_q_values)

        # actor losses
        state_list = self._get_state_list(states_ndarray)
        actor_losses = []
        for i in range(self._num_sub_agents):
            net = self._agent_nets[i]
            state = state_list[i]
            new_action, _ = net.get_actions_and_logps(state, self._exploring)  # [batch_size], [batch_size]
            cur_actions = [action for action in actions]
            cur_actions[i] = new_action
            actor_loss = -self._get_values_by_states_and_actions(states, cur_actions).mean()
            actor_losses.append(actor_loss)

        # total loss
        loss = sum(actor_losses) + self._critic_loss_coef * critic_loss

        loss_info = {
            "critic_loss": critic_loss.detach().cpu().numpy(),
            "actor_losses": [loss.detach().cpu().numpy() for loss in actor_losses],
            "loss": loss.detach().cpu().numpy() if explicit_grad else loss
        }
        if explicit_grad:
            loss_info["actor_grads"] = [net.get_gradients(loss) for net in self._agent_nets]
            loss_info["critic_grad"] = self._critic_net.get_gradients(loss)

        return loss_info

    def data_parallel(self, *args, **kwargs) -> None:
        pass  # TODO

    def learn_with_data_parallel(self, batch: dict, worker_id_list: list) -> None:
        pass  # TODO

    def update(self, loss_info_list: List[dict]) -> None:
        for i, net in enumerate(self._agent_nets):
            net.apply_gradients(average_grads([loss_info["actor_grads"][i] for loss_info in loss_info_list]))
        self._critic_net.apply_gradients(average_grads([loss_info["critic_grad"] for loss_info in loss_info_list]))

    def learn(self, batch: dict) -> None:
        for _ in range(self._grad_iters):
            loss = self.get_batch_loss(batch)["loss"]
            for net in self._agent_nets:
                net.step(loss)
            self._critic_net.step(loss)

    def improve(self) -> None:
        self.learn(self._get_batch())

    def get_state(self) -> object:
        return [net.get_state() for net in self._agent_nets]

    def set_state(self, policy_state: object) -> None:
        assert isinstance(policy_state, list)
        for net, state in zip(self._agent_nets, policy_state):
            net.set_state(state)

    def load(self, path: str) -> None:
        self.set_state(torch.load(path))

    def save(self, path: str) -> None:
        torch.save(self.get_state(), path)
