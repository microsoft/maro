# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Iterator, Optional

import torch
from gym import spaces
from tianshou.data import Batch
from torch.nn import Parameter
from torch.optim import Optimizer

from maro.rl_v31.model.base import BaseNet
from maro.rl_v31.utils import convert_ndarray_to_tensor


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

    def soft_update(self, other: BaseDLPolicy, tau: float) -> None:
        assert self.__class__ == other.__class__, (
            f"Soft update can only be done between same classes. Current model type: {self.__class__}, "
            f"other model type: {other.__class__}"
        )

        for params, other_params in zip(self.parameters(), other.parameters()):
            params.data = (1 - tau) * params.data + tau * other_params.data


class BaseRLPolicy(BaseDLPolicy, metaclass=ABCMeta):
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
            optim=optim,
        )

        self.is_exploring = False

    def switch_explore(self, explore: bool) -> None:
        self.is_exploring = explore
