# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass

from maro.rl.training.algorithms.base import ACBasedParams, ACBasedTrainer


@dataclass
class ActorCriticParams(ACBasedParams):
    """Identical to `ACBasedParams`. Please refer to the doc string of `ACBasedParams`
    for detailed information.
    """

    def __post_init__(self) -> None:
        assert self.clip_ratio is None


class ActorCriticTrainer(ACBasedTrainer):
    """Actor-Critic algorithm with separate policy and value models.

    Reference:
        https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/vpg
    """

    def __init__(
        self,
        name: str,
        params: ActorCriticParams,
        replay_memory_capacity: int = 10000,
        batch_size: int = 128,
        data_parallelism: int = 1,
        reward_discount: float = 0.9,
    ) -> None:
        super(ActorCriticTrainer, self).__init__(
            name,
            params,
            replay_memory_capacity,
            batch_size,
            data_parallelism,
            reward_discount,
        )
