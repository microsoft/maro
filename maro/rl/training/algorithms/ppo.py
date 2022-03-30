# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import Dict

from maro.rl.training.algorithms.base import DiscreteACBasedParams, DiscreteACBasedTrainer, DiscretePPOBasedOps
from maro.rl.training import AbsTrainOps

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
            "device": self.device,
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
    """
    def __init__(self, name: str, params: DiscretePPOParams) -> None:
        super(DiscretePPOTrainer, self).__init__(name, params)

    def train_step(self) -> None:
        assert isinstance(self._ops, DiscretePPOBasedOps)
        batch = self._get_batch()
        for _ in range(self._params.grad_iters):
            self._ops.update_critic(batch)
            self._ops.update_actor(batch)
        self._ops._policy_old.set_state(self._ops._policy.get_state())

    def get_local_ops_by_name(self, name: str) -> AbsTrainOps:
        return DiscretePPOBasedOps(
            name=name, get_policy_func=self._get_policy_func, parallelism=self._params.data_parallelism,
            **self._params.extract_ops_params(),
        )