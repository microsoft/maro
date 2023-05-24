# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Any, Optional

import torch
from gym import spaces
from tianshou.data import Batch
from torch import nn
from torch.optim import Optimizer

from maro.rl_v31.exploration.strategy import ExploreStrategy
from maro.rl_v31.model.qnet import DiscreteQNet
from maro.rl_v31.policy.base import BaseRLPolicy
from maro.rl_v31.utils import to_torch


class DQNPolicy(BaseRLPolicy):
    def __init__(
        self,
        name: str,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        optim: Optimizer,
        qnet: DiscreteQNet,
        explore_strategy: Optional[ExploreStrategy] = None,
    ) -> None:
        super().__init__(
            name=name,
            obs_space=obs_space,
            action_space=action_space,
            optim=optim,
        )

        self._qnet = qnet
        self._explore_strategy = explore_strategy

        assert isinstance(action_space, spaces.Discrete)
        self._action_num = action_space.n

    def q_values(self, obs: Any, act: torch.Tensor) -> torch.Tensor:  # (B,)
        return self._qnet.q_values(obs, act)

    def q_values_for_all(self, obs: Any) -> torch.Tensor:  # (B, action_num)
        return self._qnet.q_values_for_all(obs)

    def get_random_action(self, batch: Batch, **kwargs: Any) -> Batch:
        logits = torch.ones((len(batch), self._action_num)) / self._action_num  # (B, action_num)
        dist = torch.distributions.Categorical(logits)
        act = dist.sample().long()  # (B,)
        log_probs = dist.log_prob(act)  # (B,)

        # act shape: (B,)
        # probs shape: (B,)
        return Batch(act=act, probs=torch.exp(log_probs))

    def get_action(self, batch: Batch, use: str, **kwargs: Any) -> Batch:
        obs = to_torch(batch[use])
        q = self.q_values_for_all(obs)  # (B, action_num)
        q_softmax = nn.Softmax(dim=1)(q)  # (B, action_num)
        _, act = q.max(dim=1)  # (B,)

        if self.is_exploring and self._explore_strategy is not None:
            act = self._explore_strategy.get_action(obs=obs, action=act)  # (B,)
        act = act.long()

        probs = q_softmax.gather(1, act.unsqueeze(1)).squeeze(-1)  # (B,)
        return Batch(act=act, probs=probs)
