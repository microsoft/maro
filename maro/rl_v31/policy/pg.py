# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from abc import ABCMeta
from typing import Any, Callable, Type, Union, cast

import torch
from gym import spaces
from tianshou.data import Batch
from torch.distributions import Distribution
from torch.optim import Optimizer

from maro.rl_v31.model.base import PolicyModel
from maro.rl_v31.policy.base import BaseRLPolicy
from maro.rl_v31.utils import to_torch


class PGPolicy(BaseRLPolicy, metaclass=ABCMeta):
    def __init__(
        self,
        name: str,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        model: PolicyModel,
        optim: Optimizer,
        dist_fn: Union[Type[Distribution], Callable[[torch.Tensor], Distribution]] = torch.distributions.Categorical,
    ) -> None:
        super().__init__(
            name=name,
            obs_space=obs_space,
            action_space=action_space,
            optim=optim,
        )

        self.model = model
        self.dist_fn = dist_fn

    def get_random_action(self, batch: Batch, **kwargs: Any) -> Batch:
        if self.is_discrete:
            action_space = cast(spaces.Discrete, self.action_space)
            logits = torch.ones((len(batch), action_space.n)) / action_space.n  # (B, action_num)
            dist = torch.distributions.Categorical(logits)
            act = dist.sample().long()  # (B,)
        else:
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
            dist = self.dist_fn(logits)
        elif isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            raise ValueError(f"Logits of type {type(logits)} is not acceptable.")

        if self.is_discrete:
            act = dist.sample() if self.is_exploring else logits.argmax(-1)  # (B,)
            act = act.long()
        else:
            act = dist.sample()  # (B,)

        return Batch(act=act, dist=dist, logits=logits)  # TODO: do we need to return logits?


class ContinuousPGPolicy(PGPolicy):
    def __init__(
        self,
        name: str,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        model: PolicyModel,
        optim: Optimizer,
    ) -> None:
        super().__init__(
            name=name,
            obs_space=obs_space,
            action_space=action_space,
            model=model,
            optim=optim,
        )

        assert not self.is_discrete


class DiscretePGPolicy(PGPolicy):
    def __init__(
        self,
        name: str,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        model: PolicyModel,
        optim: Optimizer,
    ) -> None:
        super().__init__(
            name=name,
            obs_space=obs_space,
            action_space=action_space,
            model=model,
            optim=optim,
        )

        assert self.is_discrete
