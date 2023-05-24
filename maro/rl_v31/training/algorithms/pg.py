# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Any, Callable, Tuple, cast

import numpy as np
import torch
from tianshou.data import Batch

from maro.rl_v31.model.vnet import VCritic
from maro.rl_v31.policy.pg import PGPolicy
from maro.rl_v31.training.trainer import BaseTrainOps, SingleAgentTrainer
from maro.rl_v31.utils import discount_cumsum, to_torch

# TODO: handle ops device


class PolicyGradientOps(BaseTrainOps):
    def __init__(
        self,
        name: str,
        policy: PGPolicy,
        critic_func: Callable[[], VCritic],
        reward_discount: float = 0.99,
        critic_loss_cls: Callable = torch.nn.MSELoss,
        lam: float = 0.9,
        min_logp: float = float("-inf"),
    ) -> None:
        super().__init__(name=name, policy=policy, reward_discount=reward_discount)

        self._critic = critic_func()
        self._critic_loss_func = critic_loss_cls()
        self._lam = lam
        self._min_logp = min_logp

    def _get_auxiliary_state(self) -> dict:
        pass

    def _set_auxiliary_state(self, auxiliary_state: dict) -> None:
        pass

    def _get_actor_loss(self, batch: Batch) -> Tuple[torch.Tensor, bool]:
        # TODO: use minibatch?
        dist = self._policy(batch).dist
        act = to_torch(batch.action)
        adv = to_torch(batch.adv)
        logps = dist.log_prob(act).reshape(len(adv), -1).transpose(0, 1)

        loss = -(logps * adv).mean()
        return loss, False

    def _get_critic_loss(self, batch: Batch) -> torch.Tensor:
        # TODO: use minibatch?
        self._critic.train()
        returns = to_torch(batch.returns)
        values = self._critic(batch)
        return self._critic_loss_func(values, returns)

    def update_actor(self, batch: Batch) -> Tuple[float, bool]:
        self._policy.train()
        loss, early_stop = self._get_actor_loss(batch)
        self._policy.train_step(loss)
        return loss.detach().cpu().numpy().item(), early_stop

    def update_critic(self, batch: Batch) -> None:  # TODO: return what?
        self._critic.train()
        loss = self._get_critic_loss(batch)
        self._critic.train_step(loss)

    def preprocess_batch(self, batch: Batch) -> Batch:
        with torch.no_grad():
            self._critic.eval()
            self._policy.eval()
            values = self._critic(batch).detach().cpu().numpy()

            returns = np.zeros(len(batch), dtype=np.float32)
            adv = np.zeros(len(batch), dtype=np.float32)
            i = 0
            while i < len(batch):
                j = i
                while j < len(batch) - 1 and not (batch.terminal[j] or batch.truncated[j]):
                    j += 1

                last_val = 0.0 if batch.terminal[j] else self._critic(batch[j : j + 1], use="next_obs").item()
                cur_values = np.append(values[i : j + 1], last_val)
                cur_rewards = np.append(batch.reward[i : j + 1], last_val)
                cur_deltas = cur_rewards[:-1] + self._reward_discount * cur_values[1:] - cur_values[:-1]
                returns[i : j + 1] = discount_cumsum(cur_rewards, self._reward_discount)[:-1]
                adv[i : j + 1] = discount_cumsum(cur_deltas, self._reward_discount * self._lam)

                i = j + 1

            batch.returns = returns
            batch.adv = adv

        return batch

    def to_device(self, device: torch.device) -> None:
        self._critic.to(device)


class PolicyGradientTrainer(SingleAgentTrainer):
    def __init__(
        self,
        name: str,
        memory_size: int,
        critic_func: Callable[[], VCritic],
        batch_size: int = 128,
        reward_discount: float = 0.99,
        grad_iters: int = 1,
        critic_loss_cls: Callable = torch.nn.MSELoss,
        lam: float = 0.9,
        min_logp: float = float("-inf"),
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
        self._grad_iters = grad_iters
        self._critic_loss_cls = critic_loss_cls
        self._lam = lam
        self._min_logp = min_logp

    def build(self) -> None:
        self._ops = PolicyGradientOps(
            name=self.policy.name,
            policy=cast(PGPolicy, self.policy),
            critic_func=self._critic_func,
            reward_discount=self._reward_discount,
            critic_loss_cls=self._critic_loss_cls,
            lam=self._lam,
            min_logp=self._min_logp,
        )

    def train_step(self) -> None:
        batch_dict = self.rmm.sample(size=None, random=False, pop=True)
        batch_list = [self._ops.preprocess_batch(batch) for batch in batch_dict.values()]
        batch = Batch.cat(batch_list)
        batch.adv = (batch.adv - batch.adv.mean()) / batch.adv.std()

        for _ in range(self._grad_iters):
            early_stop = self._ops.update_actor(batch)[1]
            if early_stop:
                break

        for _ in range(self._grad_iters):
            self._ops.update_critic(batch)

    def to_device(self, device: torch.device) -> None:
        self._ops.to_device(device)
