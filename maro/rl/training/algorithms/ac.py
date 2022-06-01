# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import Dict

from maro.rl.training.algorithms.base import ACBasedParams, ACBasedTrainer


@dataclass
class ActorCriticParams(ACBasedParams):
    """Identical to `ACBasedParams`. Please refer to the doc string of `ACBasedParams`
    for detailed information.
    """

    def extract_ops_params(self) -> Dict[str, object]:
        return {
            "get_v_critic_net_func": self.get_v_critic_net_func,
            "reward_discount": self.reward_discount,
            "critic_loss_cls": self.critic_loss_cls,
            "lam": self.lam,
            "min_logp": self.min_logp,
            "is_discrete_action": self.is_discrete_action,
        }

    def __post_init__(self) -> None:
        assert self.get_v_critic_net_func is not None


class ActorCriticTrainer(ACBasedTrainer):
    """Actor-Critic algorithm with separate policy and value models.

    Reference:
        https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/vpg
    """

    def __init__(self, name: str, params: ActorCriticParams) -> None:
        super(ActorCriticTrainer, self).__init__(name, params)
