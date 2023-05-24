# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, cast

import torch
from tianshou.data import Batch

from maro.rl_v31.policy.dqn import DQNPolicy
from maro.rl_v31.training.trainer import BaseTrainOps, SingleAgentTrainer
from maro.rl_v31.utils import to_torch
from maro.utils import clone


class DQNOps(BaseTrainOps):
    def __init__(
        self,
        name: str,
        policy: DQNPolicy,
        reward_discount: float = 0.99,
        soft_update_coef: float = 0.1,
        use_prioritized_replay: bool = False,
        alpha: float = 0.4,
        beta: float = 0.6,
        double: bool = False,
    ) -> None:
        super().__init__(name=name, policy=policy, reward_discount=reward_discount)

        self._soft_update_coef = soft_update_coef
        self._use_prioritized_replay = use_prioritized_replay
        self._alpha = alpha
        self._beta = beta
        self._double = double

        self._target_policy: DQNPolicy = clone(self._policy)
        self._target_policy.name = f"target_{self._policy.name}"
        self._target_policy.eval()

    def _get_auxiliary_state(self) -> dict:
        pass

    def _set_auxiliary_state(self, auxiliary_state: dict) -> None:
        pass

    def _get_batch_loss(self, batch: Batch) -> torch.Tensor:
        act = to_torch(batch.action)
        obs = to_torch(batch.obs)
        next_obs = to_torch(batch.next_obs)
        reward = to_torch(batch.reward)
        terminal = to_torch(batch.terminal).float()

        with torch.no_grad():
            if self._double:
                self._policy.switch_explore(False)
                next_q_values = self._target_policy.q_values(next_obs, self._policy(next_obs))  # (B,)
            else:
                next_q_values = self._target_policy.q_values_for_all(next_obs).max(dim=1)[0]  # (B,)

        target_q_values = (reward + self._reward_discount * (1.0 - terminal) * next_q_values).detach()
        q_values = self._policy.q_values(obs, act)
        td_error = target_q_values - q_values

        return (td_error.pow(2)).mean()

    def update(self, batch: Batch) -> None:
        self._policy.train()
        loss = self._get_batch_loss(batch)
        self._policy.train_step(loss)

    def soft_update_target(self) -> None:
        """Soft update the target policy."""
        self._target_policy.soft_update(self._policy, self._soft_update_coef)


class DQNTrainer(SingleAgentTrainer):
    def __init__(
        self,
        name: str,
        memory_size: int,
        batch_size: int = 128,
        reward_discount: float = 0.99,
        use_prioritized_replay: bool = False,
        alpha: float = 0.4,
        beta: float = 0.6,
        num_epochs: int = 1,
        update_target_every: int = 5,
        soft_update_coef: float = 0.1,
        double: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            memory_size=memory_size,
            batch_size=batch_size,
            reward_discount=reward_discount,
            **kwargs,
        )

        self._use_prioritized_replay = use_prioritized_replay
        self._alpha = alpha
        self._beta = beta
        self._num_epochs = num_epochs
        self._update_target_every = update_target_every
        self._soft_update_coef = soft_update_coef
        self._double = double

        self._q_net_version = self._target_q_net_version = 0

    def build(self) -> None:
        self._ops = DQNOps(
            name=self.policy.name,
            policy=cast(DQNPolicy, self.policy),
            reward_discount=self._reward_discount,
            soft_update_coef=self._soft_update_coef,
            use_prioritized_replay=self._use_prioritized_replay,
            alpha=self._alpha,
            beta=self._beta,
            double=self._double,
        )

    def train_step(self) -> None:
        for _ in range(self._num_epochs):
            batch_dict = self.rmm.sample(size=self._batch_size, random=True, pop=False)
            batch_list = list(batch_dict.values())
            batch = Batch.cat(batch_list)

            self._ops.update(batch)

        self._try_soft_update_target()

    def to_device(self, device: torch.device) -> None:
        pass

    def _try_soft_update_target(self) -> None:
        """Soft update the target policy and target critic."""
        self._q_net_version += 1
        if self._q_net_version - self._target_q_net_version == self._update_target_every:
            self._ops.soft_update_target()
            self._target_q_net_version = self._q_net_version
