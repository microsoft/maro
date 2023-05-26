# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from abc import ABCMeta
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from maro.rl_v31.model.base import BaseNet


class VCritic(BaseNet, metaclass=ABCMeta):
    def __init__(self, model: nn.Module, optim: Optimizer) -> None:
        super().__init__()

        self.model = model
        self.optim = optim

    def forward(self, obs: Any, **kwargs: Any) -> torch.Tensor:  # (B,)
        critic_value = self.model(obs).float()
        assert critic_value.shape == torch.Size([len(obs)])
        return critic_value

    def train_step(self, loss: torch.Tensor) -> None:
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
