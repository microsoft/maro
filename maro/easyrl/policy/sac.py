# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable

from maro.rl.model import QNet
from maro.rl.policy import ContinuousRLPolicy
from maro.rl.training.algorithms import SoftActorCriticParams, SoftActorCriticTrainer
from .base import EasyPolicy


class SACPolicy(EasyPolicy):
    def __init__(
        self,
        actor: ContinuousRLPolicy,
        critic: QNet,
        *,
        replay_memory_capacity: int = 10000,
        batch_size: int = 128,
        reward_discount: float = 0.9,
        update_target_every: int = 5,
        random_overwrite: bool = False,
        entropy_coef: float = 0.1,
        num_epochs: int = 1,
        n_start_train: int = 0,
        q_value_loss_cls: Callable = None,
        soft_update_coef: float = 1.0,
    ) -> None:
        assert isinstance(actor, ContinuousRLPolicy)
        assert isinstance(critic, QNet)

        trainer = SoftActorCriticTrainer(
            name=actor.name,
            params=SoftActorCriticParams(
                replay_memory_capacity=replay_memory_capacity,
                batch_size=batch_size,
                reward_discount=reward_discount,
                get_q_critic_net_func=lambda: critic,
                update_target_every=update_target_every,
                random_overwrite=random_overwrite,
                entropy_coef=entropy_coef,
                num_epochs=num_epochs,
                n_start_train=n_start_train,
                q_value_loss_cls=q_value_loss_cls,
                soft_update_coef=soft_update_coef,
            ),
        )
        super(SACPolicy, self).__init__(actor, trainer)
