# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Any, Callable, Type, Union

import torch
from gym import spaces
from tianshou.data import Batch
from torch.optim import Optimizer
from torch.distributions import Distribution
from maro.rl_v31.model.base import PolicyModel
from maro.rl_v31.policy.base import BaseRLPolicy
from maro.rl_v31.utils import to_torch


class PGPolicy(BaseRLPolicy):
    def __init__(
        self,
        name: str,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        model: PolicyModel,
        optim: Optimizer,
        is_discrete: bool,
        dist_fn: Union[Type[Distribution], Callable[[torch.Tensor], Distribution]] = torch.distributions.Categorical,
    ) -> None:
        assert isinstance(action_space, (spaces.Box, spaces.Discrete))

        super().__init__(
            name=name,
            obs_space=obs_space,
            action_space=action_space,
            optim=optim,
        )

        self.model = model
        self.is_discrete = is_discrete
        self.dist_fn = dist_fn

    def forward(self, batch: Batch, **kwargs: Any) -> Batch:
        obs = to_torch(batch.obs)
        logits = self.model(obs)
        dist = self.dist_fn(logits)

        if self.is_discrete:
            act = dist.sample() if self.is_exploring else logits.argmax(-1)
        else:
            act = logits

        return Batch(act=act, dist=dist, logits=logits)


class ContinuousPGPolicy(PGPolicy):
    def __init__(
        self,
        name: str,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        model: PolicyModel,
        optim: Optimizer,
    ) -> None:
        assert isinstance(action_space, spaces.Box), "Action space of ContinuousPGPolicy should be `spaces.Box`."

        super().__init__(
            name=name,
            obs_space=obs_space,
            action_space=action_space,
            model=model,
            optim=optim,
            is_discrete=False,
        )


class DiscretePGPolicy(PGPolicy):
    def __init__(
        self,
        name: str,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        model: PolicyModel,
        optim: Optimizer,
    ) -> None:
        assert isinstance(
            action_space,
            spaces.Discrete,
        ), "Action space of DiscretePGPolicy should be `spaces.Discrete`."

        super().__init__(
            name=name,
            obs_space=obs_space,
            action_space=action_space,
            model=model,
            optim=optim,
            is_discrete=True,
        )
