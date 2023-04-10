# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Any, Callable, cast, Tuple

import torch
from tianshou.data import Batch

from maro.rl_v31.model.vnet import VCritic
from maro.rl_v31.policy.pg import PGPolicy
from maro.rl_v31.training.algorithms.pg import PolicyGradientTrainer, PolicyGradientTrainOps
from maro.rl_v31.training.replay_memory import ReplayMemoryManager
from maro.rl_v31.utils import to_torch


class PPOTrainOps(PolicyGradientTrainOps):
    def __init__(
        self,
        name: str,
        policy: PGPolicy,
        critic_func: Callable[[], VCritic],
        clip_ratio: float,
        reward_discount: float = 0.99,
        critic_loss_cls: Callable = torch.nn.MSELoss,
        lam: float = 0.9,
        min_logp: float = float("-inf"),
    ) -> None:
        super().__init__(
            name=name,
            policy=policy,
            critic_func=critic_func,
            reward_discount=reward_discount,
            critic_loss_cls=critic_loss_cls,
            lam=lam,
            min_logp=min_logp,
        )

        self._clip_ratio = clip_ratio

    def _get_actor_loss(self, batch: Batch) -> Tuple[torch.Tensor, bool]:
        # TODO: use minibatch?
        dist = self._policy(batch).dist
        act = to_torch(batch.action)
        adv = to_torch(batch.adv)
        logps_old = to_torch(batch.logps_old)
        if self._policy.is_discrete:
            logps = dist.log_prob(act).reshape(len(adv), -1).transpose(0, 1).squeeze()
        else:
            logps = dist.log_prob(act).sum(axis=-1)

        ratio = torch.exp(logps - logps_old)
        kl = (logps_old - logps).mean().item()
        early_stop = kl >= 0.01 * 1.5  # TODO
        clipped_ratio = torch.clamp(ratio, 1 - self._clip_ratio, 1 + self._clip_ratio)
        loss = -(torch.min(ratio * adv, clipped_ratio * adv)).mean()

        return loss, early_stop

    def preprocess_batch(self, batch: Batch) -> Batch:
        batch = super().preprocess_batch(batch)

        with torch.no_grad():
            dist = self._policy(batch).dist
            act = to_torch(batch.action)
            adv = to_torch(batch.adv)
            if self._policy.is_discrete:
                logps = dist.log_prob(act).reshape(len(adv), -1).transpose(0, 1).squeeze()
            else:
                logps = dist.log_prob(act).sum(axis=-1)
            batch.logps_old = logps

        return batch


class PPOTrainer(PolicyGradientTrainer):
    def __init__(
        self,
        name: str,
        rmm: ReplayMemoryManager,
        critic_func: Callable[[], VCritic],
        clip_ratio: float = 1.0,
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
            rmm=rmm,
            critic_func=critic_func,
            batch_size=batch_size,
            reward_discount=reward_discount,
            grad_iters=grad_iters,
            critic_loss_cls=critic_loss_cls,
            lam=lam,
            min_logp=min_logp,
            **kwargs,
        )

        self._clip_ratio = clip_ratio

    def build(self) -> None:
        self._ops = PPOTrainOps(
            name=self.policy.name,
            policy=cast(PGPolicy, self.policy),
            critic_func=self._critic_func,
            reward_discount=self._reward_discount,
            critic_loss_cls=self._critic_loss_cls,
            lam=self._lam,
            min_logp=self._min_logp,
            clip_ratio=self._clip_ratio,
        )
