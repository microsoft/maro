# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import Dict

from maro.rl.training.algorithms.base import DiscreteACBasedParams, DiscreteACBasedTrainer


@dataclass
class DiscreteActorCriticParams(DiscreteACBasedParams):
    """Identical to `DiscreteACBasedParams`. Please refer to the doc string of `DiscreteACBasedParams`
    for detailed information.
    """
    def extract_ops_params(self) -> Dict[str, object]:
        return {
            "get_v_critic_net_func": self.get_v_critic_net_func,
            "reward_discount": self.reward_discount,
            "critic_loss_cls": self.critic_loss_cls,
            "lam": self.lam,
            "min_logp": self.min_logp,
        }

    def __post_init__(self) -> None:
        assert self.get_v_critic_net_func is not None


class DiscreteActorCriticTrainer(DiscreteACBasedTrainer):
    """Actor Critic algorithm with separate policy and value models.
    """
    def __init__(self, name: str, params: DiscreteActorCriticParams) -> None:
        super(DiscreteActorCriticTrainer, self).__init__(name, params)
