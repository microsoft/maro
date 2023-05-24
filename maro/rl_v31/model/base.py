# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Tuple, Union

import torch
from torch import nn


class BaseNet(nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass
    
    def soft_update(self, other: BaseNet, tau: float) -> None:
        assert self.__class__ == other.__class__, (
            f"Soft update can only be done between same classes. Current model type: {self.__class__}, "
            f"other model type: {other.__class__}"
        )

        for params, other_params in zip(self.parameters(), other.parameters()):
            params.data = (1 - tau) * params.data + tau * other_params.data


class PolicyModel(BaseNet):
    @abstractmethod
    def forward(self, obs: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        raise NotImplementedError
