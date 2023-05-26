# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Any, Callable, cast, Tuple

import torch
from tianshou.data import Batch

from maro.rl_v31.model.qnet import QCritic
from maro.rl_v31.policy.sac import SACPolicy
from maro.rl_v31.training.trainer import BaseTrainOps, SingleAgentTrainer
from maro.rl_v31.utils import to_torch
from maro.utils import clone


class SACOps(BaseTrainOps):
    def __init__(
        self,
        name: str,
        policy: SACPolicy,
        critic_func: Callable[[], QCritic],
        reward_discount: float = 0.99,
        entropy_coef: float = 0.1,
        soft_update_coef: float = 0.05
    ) -> None:
        super().__init__(name=name, policy=policy, reward_discount=reward_discount)

        self._q_critic1 = critic_func()
        self._q_critic2 = critic_func()
        self._target_q_critic1: QCritic = clone(self._q_critic1)
        self._target_q_critic2: QCritic = clone(self._q_critic2)
        self._target_q_critic1.eval()
        self._target_q_critic2.eval()
        self._q_value_loss_func = torch.nn.MSELoss()

        self._entropy_coef = entropy_coef
        self._soft_update_coef = soft_update_coef

    def _get_critic_loss(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        self._q_critic1.train()
        self._q_critic2.train()
        obs = to_torch(batch.obs)
        next_obs = to_torch(batch.next_obs)
        act = to_torch(batch.action)
        reward = to_torch(batch.reward)
        terminal = to_torch(batch.terminal).float()

        with torch.no_grad():
            t = self._policy(batch, use="next_obs")
            next_act = t.act
            next_logps = t.logps
            target_q1 = self._target_q_critic1(next_obs, next_act)
            target_q2 = self._target_q_critic2(next_obs, next_act)
            target_q = torch.min(target_q1, target_q2)
            y = (reward + self._reward_discount * (1.0 - terminal) * (target_q - self._entropy_coef * next_logps)).float()

        q1 = self._q_critic1(obs, act)
        q2 = self._q_critic2(obs, act)
        loss_q1 = self._q_value_loss_func(q1, y)
        loss_q2 = self._q_value_loss_func(q2, y)
        return loss_q1, loss_q2

    def update_critic(self, batch: Batch) -> Tuple[float, float]:
        self._q_critic1.train()
        self._q_critic2.train()
        loss_q1, loss_q2 = self._get_critic_loss(batch)
        self._q_critic1.train_step(loss_q1)
        self._q_critic2.train_step(loss_q2)
        return loss_q1.detach().cpu().numpy().item(), loss_q2.detach().cpu().numpy().item()

    def _get_actor_loss(self, batch: Batch) -> Tuple[torch.Tensor, bool]:
        self._q_critic1.freeze()
        self._q_critic2.freeze()
        self._policy.train()

        obs = to_torch(batch.obs)
        t = self._policy(batch)
        act = t.act
        logps = t.logps
        q1 = self._q_critic1(obs, act)
        q2 = self._q_critic2(obs, act)
        q = torch.min(q1, q2)

        loss = (self._entropy_coef * logps - q).mean()

        self._q_critic1.unfreeze()
        self._q_critic2.unfreeze()

        return loss, False  # TODO: always False?

    def update_actor(self, batch: Batch) -> Tuple[float, bool]:
        self._policy.train()
        loss, early_stop = self._get_actor_loss(batch)
        self._policy.train_step(loss)
        return loss.detach().cpu().numpy().item(), early_stop

    def soft_update_target(self) -> None:
        self._target_q_critic1.soft_update(self._q_critic1, self._soft_update_coef)
        self._target_q_critic2.soft_update(self._q_critic2, self._soft_update_coef)

    def _get_auxiliary_state(self) -> dict:
        pass

    def _set_auxiliary_state(self, auxiliary_state: dict) -> None:
        pass


class SACTrainer(SingleAgentTrainer):
    def __init__(
        self,
        name: str,
        memory_size: int,
        critic_func: Callable[[], QCritic],
        batch_size: int = 128,
        reward_discount: float = 0.99,
        entropy_coef: float = 0.1,
        soft_update_coef: float = 0.05,
        update_target_every: int = 5,
        n_start_train: int = 0,
        num_epochs: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            memory_size=memory_size,
            batch_size=batch_size,
            reward_discount=reward_discount,
            **kwargs,
        )

        self._critic_func = critic_func
        self._entropy_coef = entropy_coef
        self._soft_update_coef = soft_update_coef
        self._policy_version = self._target_policy_version = 0
        self._update_target_every = update_target_every
        self._n_start_train = n_start_train
        self._num_epochs = num_epochs

    def train_step(self) -> None:
        if self.rmm.n_sample < self._n_start_train:
            return

        for _ in range(self._num_epochs):
            batch_dict = self.rmm.sample(size=self._batch_size, random=True, pop=False)
            batch_list = list(batch_dict.values())
            batch = Batch.cat(batch_list)

            self._ops.update_critic(batch)
            self._ops.update_actor(batch)
            self._try_soft_update_target()

    def build(self) -> None:
        self._ops = SACOps(
            name=self.policy.name,
            policy=cast(SACPolicy, self.policy),
            critic_func=self._critic_func,
            reward_discount=self._reward_discount,
            entropy_coef=self._entropy_coef,
            soft_update_coef=self._soft_update_coef,
        )

    def to_device(self, device: torch.device) -> None:
        pass

    def _try_soft_update_target(self) -> None:
        """Soft update the target policy and target critic."""
        self._policy_version += 1
        if self._policy_version - self._target_policy_version == self._update_target_every:
            self._ops.soft_update_target()
            self._target_policy_version = self._policy_version
