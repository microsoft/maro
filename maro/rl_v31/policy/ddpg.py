# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Any, Optional, cast

import torch
from gym import spaces
from tianshou.data import Batch
from torch.optim import Optimizer

from maro.rl_v31.exploration import ExploreStrategy
from maro.rl_v31.model.base import PolicyModel
from maro.rl_v31.policy.base import BaseRLPolicy
from maro.rl_v31.utils import to_torch


class DDPGPolicy(BaseRLPolicy):
    def __init__(
        self,
        name: str,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        model: PolicyModel,
        optim: Optimizer,
        explore_strategy: Optional[ExploreStrategy] = None,
    ) -> None:
        super().__init__(
            name=name,
            obs_space=obs_space,
            action_space=action_space,
            optim=optim,
        )

        assert not self.is_discrete

        self.model = model
        self._explore_strategy = explore_strategy

    def get_random_action(self, batch: Batch, **kwargs: Any) -> Batch:
        action_space = cast(spaces.Box, self.action_space)
        low = torch.Tensor(action_space.low).repeat(len(batch), 1)  # (B, action_dim)
        high = torch.Tensor(action_space.high).repeat(len(batch), 1)  # (B, action_dim)
        dist = torch.distributions.Uniform(low, high)
        act = dist.sample()  # (B, action_dim)
        return Batch(act=act)

    def get_action(self, batch: Batch, use: str, **kwargs: Any) -> Batch:
        obs = to_torch(batch[use])
        act = self.model(obs)

        if self.is_exploring and self._explore_strategy is not None:
            act = self._explore_strategy.get_action(obs=obs, action=act)

        return Batch(act=act)
