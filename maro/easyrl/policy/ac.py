# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable, Optional

from maro.rl.model import VNet
from maro.rl.policy import GradientPolicy
from maro.rl.training.algorithms import ActorCriticParams, ActorCriticTrainer
from .base import EasyPolicy


class A2CPolicy(EasyPolicy):
    def __init__(
        self,
        actor: GradientPolicy,
        critic: VNet,
        *,
        replay_memory_capacity: int = 10000,
        batch_size: int = 128,
        reward_discount: float = 0.9,
        grad_iters: int = 1,
        critic_loss_cls: Callable = None,
        lam: float = 0.9,
        min_logp: Optional[float] = None,
    ) -> None:
        trainer = ActorCriticTrainer(
            name=actor.name,
            params=ActorCriticParams(
                replay_memory_capacity=replay_memory_capacity,
                batch_size=batch_size,
                get_v_critic_net_func=lambda: critic,
                reward_discount=reward_discount,
                grad_iters=grad_iters,
                critic_loss_cls=critic_loss_cls,
                lam=lam,
                min_logp=min_logp,
                is_discrete_action=actor.is_discrete_action,
            )
        )
        super(A2CPolicy, self).__init__(actor, trainer)
