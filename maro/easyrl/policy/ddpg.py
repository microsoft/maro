# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable

from maro.rl.model import QNet
from maro.rl.policy import ContinuousRLPolicy
from maro.rl.training.algorithms import DDPGParams, DDPGTrainer
from .base import EasyPolicy


class DDPGPolicy(EasyPolicy):
    def __init__(
        self,
        actor: ContinuousRLPolicy,
        critic: QNet,
        *,
        replay_memory_capacity: int = 10000,
        batch_size: int = 128,
        reward_discount: float = 0.9,
        num_epochs: int = 1,
        update_target_every: int = 5,
        q_value_loss_cls: Callable = None,
        soft_update_coef: float = 1.0,
        random_overwrite: bool = False,
        min_num_to_trigger_training: int = 0,
    ) -> None:
        assert isinstance(actor, ContinuousRLPolicy)
        assert isinstance(critic, QNet)

        trainer = DDPGTrainer(
            name=actor.name,
            params=DDPGParams(
                replay_memory_capacity=replay_memory_capacity,
                batch_size=batch_size,
                reward_discount=reward_discount,
                get_q_critic_net_func=lambda: critic,
                num_epochs=num_epochs,
                update_target_every=update_target_every,
                q_value_loss_cls=q_value_loss_cls,
                soft_update_coef=soft_update_coef,
                random_overwrite=random_overwrite,
                min_num_to_trigger_training=min_num_to_trigger_training,
            ),
        )
        super(DDPGPolicy, self).__init__(actor, trainer)
