# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
from typing import List, Tuple, cast

import dgl
import torch

from maro.rl.model import DiscretePolicyNet
from maro.rl.policy import DiscretePolicyGradient, RLPolicy
from maro.rl.training.algorithms.base import ACBasedOps, ACBasedParams, ACBasedTrainer
from maro.rl.utils import ndarray_to_tensor
from maro.utils import LogFormat, Logger

from examples.mis.lwd.ppo.replay_memory import BatchGraphReplayMemory, GraphBasedExpElement, GraphBasedTransitionBatch


class GraphBasedPPOTrainOps(ACBasedOps):
    def __init__(
        self,
        name: str,
        policy: RLPolicy,
        params: ACBasedParams,
        reward_discount: float,
        parallelism: int = 1,
    ) -> None:
        super().__init__(name, policy, params, reward_discount, parallelism)

        self._clip_lower_bound = math.log(1.0 - self._clip_ratio)
        self._clip_upper_bound = math.log(1.0 + self._clip_ratio)

    def _get_critic_loss(self, batch: GraphBasedTransitionBatch) -> torch.Tensor:
        states = ndarray_to_tensor(batch.states, device=self._device).permute(1, 0, 2)
        returns = ndarray_to_tensor(batch.returns, device=self._device)

        self._v_critic_net.train()
        kwargs = {"graph": batch.graph}
        value_preds = self._v_critic_net.v_values(states, **kwargs).permute(1, 0)
        critic_loss = 0.5 * (value_preds - returns).pow(2).mean()
        return critic_loss

    def _get_actor_loss(self, batch: GraphBasedTransitionBatch) -> Tuple[torch.Tensor, bool]:
        graph = batch.graph
        kwargs = {"graph": batch.graph}

        states = ndarray_to_tensor(batch.states, device=self._device).permute(1, 0, 2)
        actions = ndarray_to_tensor(batch.actions, device=self._device).permute(1, 0)
        advantages = ndarray_to_tensor(batch.advantages, device=self._device)
        logps_old = ndarray_to_tensor(batch.old_logps, device=self._device)
        if self._is_discrete_action:
            actions = actions.long()

        self._policy.train()
        logps = self._policy.get_states_actions_logps(states, actions, **kwargs).permute(1, 0)
        diff = logps - logps_old
        clamped_diff = torch.clamp(diff, self._clip_lower_bound, self._clip_upper_bound)
        stacked_diff = torch.stack([diff, clamped_diff], dim=2)

        graph.ndata["h"] = stacked_diff.permute(1, 0, 2)
        h = dgl.sum_nodes(graph, "h").permute(1, 0, 2)
        graph.ndata.pop("h")
        ratio = torch.exp(h.select(2, 0))
        clamped_ratio = torch.exp(h.select(2, 1))

        actor_loss = -torch.min(ratio * advantages, clamped_ratio * advantages).mean()
        return actor_loss, False  # TODO: add early-stop logic if needed


class GraphBasedPPOTrainer(ACBasedTrainer):
    def __init__(
        self,
        name: str,
        params: ACBasedParams,
        replay_memory_capacity: int = 32,
        batch_size: int = 16,
        data_parallelism: int = 1,
        reward_discount: float = 0.9,
        graph_batch_size: int = 4,
        graph_num_samples: int = 2,
        input_feature_size: int = 2,
        num_train_epochs: int = 4,
        log_dir: str = None,
    ) -> None:
        super().__init__(name, params, replay_memory_capacity, batch_size, data_parallelism, reward_discount)

        self._graph_batch_size = graph_batch_size
        self._graph_num_samples = graph_num_samples
        self._input_feature_size = input_feature_size
        self._num_train_epochs = num_train_epochs

        self._trainer_logger = None
        if log_dir is not None:
            self._trainer_logger = Logger(
                tag=self.name,
                format_=LogFormat.none,
                dump_folder=log_dir,
                dump_mode="w",
                extension_name="csv",
                auto_timestamp=False,
                stdout_level="INFO",
            )
            self._trainer_logger.debug(
                "Steps,Mean Reward,Mean Return,Mean Advantage,0-Action,1-Action,2-Action,Critic Loss,Actor Loss"
            )

    def build(self) -> None:
        self._ops = cast(GraphBasedPPOTrainOps, self.get_ops())
        self._replay_memory = BatchGraphReplayMemory(
            max_t=self._replay_memory_capacity,
            graph_batch_size=self._graph_batch_size,
            num_samples=self._graph_num_samples,
            feature_size=self._input_feature_size,
        )

    def record_multiple(self, env_idx: int, exp_elements: List[GraphBasedExpElement]) -> None:
        self._replay_memory.reset()

        self._ops._v_critic_net.eval()
        self._ops._policy.eval()

        for exp in exp_elements:
            state = ndarray_to_tensor(exp.state, self._ops._device)
            action = ndarray_to_tensor(exp.action, self._ops._device)
            value_pred = self._ops._v_critic_net.v_values(state, graph=exp.graph).cpu().detach().numpy()
            logps = self._ops._policy.get_states_actions_logps(state, action, graph=exp.graph).cpu().detach().numpy()
            self._replay_memory.add_transition(exp, value_pred, logps)

        self._ops._v_critic_net.train()
        self._ops._policy.train()

    def get_local_ops(self) -> GraphBasedPPOTrainOps:
        return GraphBasedPPOTrainOps(
            name=self._policy.name,
            policy=self._policy,
            parallelism=self._data_parallelism,
            reward_discount=self._reward_discount,
            params=self._params,
        )

    def train_step(self) -> None:
        assert isinstance(self._ops, GraphBasedPPOTrainOps)

        data_loader = self._replay_memory.build_update_sampler(
            batch_size=self._batch_size,
            num_train_epochs=self._num_train_epochs,
            gamma=self._reward_discount,
        )

        statistics = self._replay_memory.get_statistics()

        avg_critic_loss, avg_actor_loss = 0, 0
        for batch in data_loader:
            for _ in range(self._params.grad_iters):
                critic_loss = self._ops.update_critic(batch)
                actor_loss, _ = self._ops.update_actor(batch)
                avg_critic_loss += critic_loss
                avg_actor_loss += actor_loss
        avg_critic_loss /= self._params.grad_iters
        avg_actor_loss /= self._params.grad_iters

        if self._trainer_logger is not None:
            self._trainer_logger.debug(
                f"{statistics['step_t']},"
                f"{statistics['reward']},"
                f"{statistics['return']},"
                f"{statistics['advantage']},"
                f"{statistics['action_0']},"
                f"{statistics['action_1']},"
                f"{statistics['action_2']},"
                f"{avg_critic_loss},"
                f"{avg_actor_loss}"
            )


class GraphBasedPPOPolicy(DiscretePolicyGradient):
    def __init__(self, name: str, policy_net: DiscretePolicyNet, trainable: bool = True, warmup: int = 0) -> None:
        super(GraphBasedPPOPolicy, self).__init__(name, policy_net, trainable, warmup)

    def get_actions_tensor(self, states: torch.Tensor, **kwargs) -> torch.Tensor:
        actions = self._get_actions_impl(states, **kwargs)
        return actions

    def get_states_actions_logps(self, states: torch.Tensor, actions: torch.Tensor, **kwargs) -> torch.Tensor:
        logps = self._get_states_actions_logps_impl(states, actions, **kwargs)
        return logps
