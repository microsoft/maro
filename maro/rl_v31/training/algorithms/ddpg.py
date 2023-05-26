from typing import Any, Callable, Tuple, cast

import torch.nn
from tianshou.data import Batch

from maro.rl_v31.model.qnet import QCritic
from maro.rl_v31.policy import DDPGPolicy
from maro.rl_v31.training.trainer import BaseTrainOps, SingleAgentTrainer
from maro.rl_v31.utils import to_torch
from maro.utils import clone


class DDPGOps(BaseTrainOps):
    def __init__(
        self,
        name: str,
        policy: DDPGPolicy,
        critic_func: Callable[[], QCritic],
        reward_discount: float = 0.99,
        soft_update_coef: float = 0.1,
    ) -> None:
        super().__init__(name=name, policy=policy, reward_discount=reward_discount)

        self._target_policy: DDPGPolicy = clone(self._policy)
        self._target_policy.name = f"target_{self._policy.name}"
        self._target_policy.eval()
        self._q_critic = critic_func()
        self._target_q_critic: QCritic = clone(self._q_critic)
        self._target_q_critic.eval()

        assert not self._target_policy.is_exploring

        self._q_loss_func = torch.nn.MSELoss()
        self._soft_update_coef = soft_update_coef

    def _get_critic_loss(self, batch: Batch) -> torch.Tensor:
        self._q_critic.train()
        obs = to_torch(batch.obs)
        next_obs = to_torch(batch.next_obs)
        act = to_torch(batch.action)
        reward = to_torch(batch.reward)
        terminal = to_torch(batch.terminal).float()

        with torch.no_grad():
            next_q_values = self._target_q_critic(
                obs=next_obs,  # s'
                act=self._target_policy(batch, use="next_obs").act,  # miu_targ(s')
            )  # Q_targ(s', miu_targ(s'))
            # y(r, s', d) = r + gamma * (1 - d) * Q_targ(s', miu_targ(s'))
            target_q_values = (reward + self._reward_discount * (1.0 - terminal) * next_q_values).detach().float()

        q_values = self._q_critic(obs=obs, act=act)  # Q(s, a)
        return self._q_loss_func(q_values, target_q_values)  # MSE(Q(s, a), y(r, s', d))

    def update_critic(self, batch: Batch) -> float:
        self._q_critic.train()
        loss = self._get_critic_loss(batch)
        self._q_critic.train_step(loss)
        return loss.detach().cpu().numpy().item()

    def _get_actor_loss(self, batch: Batch) -> Tuple[torch.Tensor, bool]:
        self._policy.train()
        obs = to_torch(batch.obs)  # s
        loss = -self._q_critic(
            obs=obs,  # s
            act=self._policy(batch).act,  # miu(s)
        ).mean()  # -Q(s, miu(s))

        return loss, False  # TODO: always False?

    def update_actor(self, batch: Batch) -> Tuple[float, bool]:
        self._policy.train()
        loss, early_stop = self._get_actor_loss(batch)
        self._policy.train_step(loss)
        return loss.detach().cpu().numpy().item(), early_stop

    def soft_update_target(self) -> None:
        """Soft update the target policy."""
        self._target_policy.soft_update(self._policy, self._soft_update_coef)
        self._target_q_critic.soft_update(self._q_critic, self._soft_update_coef)

    def _get_auxiliary_state(self) -> dict:
        pass

    def _set_auxiliary_state(self, auxiliary_state: dict) -> None:
        pass


class DDPGTrainer(SingleAgentTrainer):
    def __init__(
        self,
        name: str,
        memory_size: int,
        critic_func: Callable[[], QCritic],
        batch_size: int = 128,
        update_target_every: int = 5,
        soft_update_coef: float = 0.1,
        reward_discount: float = 0.99,
        num_epochs: int = 1,
        n_start_train: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            memory_size=memory_size,
            batch_size=batch_size,
            reward_discount=reward_discount,
            **kwargs,
        )

        self._update_target_every = update_target_every
        self._critic_func = critic_func
        self._soft_update_coef = soft_update_coef
        self._num_epochs = num_epochs
        self._n_start_train = n_start_train

        self._policy_version = self._target_policy_version = 0

    def train_step(self) -> None:
        if self.rmm.n_sample < self._n_start_train:
            return

        for _ in range(self._num_epochs):
            batch = self.rmm.sample(size=self._batch_size, random=True, pop=False)
            self._ops.update_critic(batch)
            self._ops.update_actor(batch)
            self._try_soft_update_target()

    def build(self) -> None:
        self._ops = DDPGOps(
            name=self.policy.name,
            policy=cast(DDPGPolicy, self.policy),
            critic_func=self._critic_func,
            reward_discount=self._reward_discount,
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
