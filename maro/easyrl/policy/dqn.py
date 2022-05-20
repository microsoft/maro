# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl.policy import ValueBasedPolicy
from maro.rl.training.algorithms import DQNParams, DQNTrainer
from .base import EasyPolicy


class DQNPolicy(EasyPolicy):
    def __init__(
        self,
        actor: ValueBasedPolicy,
        *,
        replay_memory_capacity: int = 10000,
        batch_size: int = 128,
        reward_discount: float = 0.9,
        num_epochs: int = 1,
        update_target_every: int = 5,
        soft_update_coef: float = 0.1,
        double: bool = False,
        random_overwrite: bool = False,
    ) -> None:
        assert isinstance(actor, ValueBasedPolicy)

        trainer = DQNTrainer(
            name=actor.name,
            params=DQNParams(
                replay_memory_capacity=replay_memory_capacity,
                batch_size=batch_size,
                reward_discount=reward_discount,
                num_epochs=num_epochs,
                update_target_every=update_target_every,
                soft_update_coef=soft_update_coef,
                double=double,
                random_overwrite=random_overwrite,
            ),
        )
        super(DQNPolicy, self).__init__(actor, trainer)
