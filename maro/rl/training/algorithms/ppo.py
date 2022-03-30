# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import Dict

from maro.rl.training.algorithms.base import DiscreteACBasedParams, DiscreteACBasedTrainer


@dataclass
class DiscretePPOParams(DiscreteACBasedParams):
    """Mostly inherited from `DiscreteACBasedParams`. Please refer to the doc string of `DiscreteACBasedParams`
    for more detailed information.

    clip_ratio (float, default=None): Clip ratio in the PPO algorithm (https://arxiv.org/pdf/1707.06347.pdf).
        If it is None, the actor loss is calculated using the usual policy gradient theorem.
    """
    clip_ratio: float = None

    def extract_ops_params(self) -> Dict[str, object]:
        return {
            "get_v_critic_net_func": self.get_v_critic_net_func,
            "reward_discount": self.reward_discount,
            "critic_loss_cls": self.critic_loss_cls,
            "clip_ratio": self.clip_ratio,
            "lam": self.lam,
            "min_logp": self.min_logp,
        }

    def __post_init__(self) -> None:
        assert self.get_v_critic_net_func is not None
        assert self.clip_ratio is not None


class DiscretePPOTrainer(DiscreteACBasedTrainer):
    """Discrete PPO algorithm.

    References:
        https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ppo.
    """
    def __init__(self, name: str, params: DiscretePPOParams) -> None:
        super(DiscretePPOTrainer, self).__init__(name, params)
