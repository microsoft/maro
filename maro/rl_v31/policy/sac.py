# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any, Callable, cast, Type, Union

import torch
from gym import spaces
from tianshou.data import Batch
from torch.distributions import Distribution, Normal
from torch.optim import Optimizer

from maro.rl_v31.model.base import PolicyModel
from maro.rl_v31.policy.base import BaseRLPolicy
from maro.rl_v31.utils import to_torch
import torch.nn.functional as F
import numpy as np


class SACPolicy(BaseRLPolicy):
    def __init__(
        self,
        name: str,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        model: PolicyModel,
        optim: Optimizer,
        action_limit: float,
        dist_fn: Union[Type[Distribution], Callable[[torch.Tensor], Distribution]] = Normal,
    ) -> None:
        super().__init__(
            name=name,
            obs_space=obs_space,
            action_space=action_space,
            optim=optim,
        )

        self.model = model
        self._action_limit = action_limit
        self._dist_fn = dist_fn

    def get_random_action(self, batch: Batch, **kwargs: Any) -> Batch:
        action_space = cast(spaces.Box, self.action_space)
        low = torch.Tensor(action_space.low).repeat(len(batch), 1)  # (B, action_dim)
        high = torch.Tensor(action_space.high).repeat(len(batch), 1)  # (B, action_dim)
        dist = torch.distributions.Uniform(low, high)
        act = dist.sample()  # (B, action_dim)
        return Batch(act=act, dist=dist)  # TODO: do we need to return logits?

    def get_action(self, batch: Batch, use: str, **kwargs: Any) -> Batch:
        obs = to_torch(batch[use])
        logits = self.model(obs)

        if isinstance(logits, torch.Tensor):
            dist = self._dist_fn(logits)
            mu = logits
        elif isinstance(logits, tuple):
            dist = self._dist_fn(*logits)
            mu = logits[0]
        else:
            raise ValueError(f"Logits of type {type(logits)} is not acceptable.")

        act = dist.rsample() if self.is_exploring else mu
        logps = dist.log_prob(act).sum(axis=-1)
        logps -= (2 * (np.log(2) - act - F.softplus(-2 * act))).sum(axis=1)
        act = torch.tanh(act) * self._action_limit

        return Batch(act=act, logps=logps)
