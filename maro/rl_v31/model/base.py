# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from abc import abstractmethod
from typing import Any

import torch
from torch import nn


class BaseNet(nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass


class PolicyModel(BaseNet):
    @abstractmethod
    def forward(self, obs: Any) -> torch.Tensor:
        raise NotImplementedError
