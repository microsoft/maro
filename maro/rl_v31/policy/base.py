# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Iterator, Optional, cast

import torch
from gym import spaces
from tianshou.data import Batch
from torch.nn import Parameter
from torch.optim import Optimizer

from maro.rl_v31.model.base import BaseNet


def _set_requires_grad(params: Iterator[Parameter], value: bool) -> None:
    for p in params:
        p.requires_grad = value


class AbsPolicy(BaseNet, metaclass=ABCMeta):
    def __init__(
        self,
        name: str,
        obs_space: spaces.Space,
        action_space: spaces.Space,
    ) -> None:
        super().__init__()

        self.name = name
        self.obs_space = obs_space
        self.action_space = action_space

        self.device: Optional[torch.device] = None

    @abstractmethod
    def forward(
        self,
        batch: Batch,
        **kwargs: Any,
    ) -> Batch:
        raise NotImplementedError

    @abstractmethod
    def get_states(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def set_states(self, state_dict: dict) -> None:
        raise NotImplementedError

    def to_device(self, device: torch.device) -> None:
        """Assign the current policy to a specific device.

        Args:
            device (torch.device): The target device.
        """
        self.device = device
        self.to(self.device)


class BaseDLPolicy(AbsPolicy, metaclass=ABCMeta):
    def __init__(
        self,
        name: str,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        optim: Optimizer,
    ) -> None:
        super().__init__(
            name=name,
            obs_space=obs_space,
            action_space=action_space,
        )

        self.optim = optim

    def get_states(self) -> dict:
        return {
            "network": self.state_dict(),
            "optim": self.optim.state_dict(),
        }

    def set_states(self, state_dict: dict) -> None:
        self.load_state_dict(state_dict["network"])
        self.optim.load_state_dict(state_dict["optim"])

    def train_step(self, loss: torch.Tensor) -> None:
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def get_gradients(self, loss) -> Dict[str, torch.Tensor]:
        self.optim.zero_grad()
        loss.backward()
        return {name: param.grad for name, param in self.named_parameters()}

    def apply_gradients(self, grad: Dict[str, torch.Tensor]) -> None:
        for name, param in self.named_parameters():
            param.grad = grad[name]
        self.optim.step()

    def freeze(self, params: Optional[Iterator[Parameter]] = None) -> None:
        _set_requires_grad(params or self.parameters(), False)

    def unfreeze(self, params: Optional[Iterator[Parameter]] = None) -> None:
        _set_requires_grad(params or self.parameters(), True)


class BaseRLPolicy(BaseDLPolicy, metaclass=ABCMeta):
    def __init__(
        self,
        name: str,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        optim: Optimizer,
        warmup: int = 0,
    ) -> None:
        super().__init__(
            name=name,
            obs_space=obs_space,
            action_space=action_space,
            optim=optim,
        )

        assert isinstance(action_space, (spaces.Box, spaces.Discrete))

        self.warmup = warmup
        self.is_discrete = isinstance(action_space, spaces.Discrete)

        self.call_count = 0
        self.is_exploring = False

    def forward(self, batch: Batch, use: str = "obs", **kwargs: Any) -> Batch:
        random = self.call_count < self.warmup
        res = self.get_random_action(batch, **kwargs) if random else self.get_action(batch, use, **kwargs)
        self.call_count += len(batch)

        act = res.act
        if self.is_discrete:
            assert act.shape == (len(batch),)
            res.act = act.long()
        else:
            action_space = cast(spaces.Box, self.action_space)
            assert act.shape == (len(batch), len(action_space.low))
            res.act = act.float()

        return res

    @abstractmethod
    def get_random_action(self, batch: Batch, **kwargs: Any) -> Batch:
        raise NotImplementedError

    @abstractmethod
    def get_action(self, batch: Batch, use: str, **kwargs: Any) -> Batch:
        raise NotImplementedError

    def switch_explore(self, explore: bool) -> None:
        self.is_exploring = explore
