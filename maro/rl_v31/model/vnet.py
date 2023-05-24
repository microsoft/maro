# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from abc import ABCMeta
from typing import Any

import torch
from tianshou.data import Batch
from torch import nn
from torch.optim import Optimizer

from maro.rl_v31.model.base import BaseNet
from maro.rl_v31.utils import to_torch


class VCritic(BaseNet, metaclass=ABCMeta):
    def __init__(self, model: nn.Module, optim: Optimizer) -> None:
        super().__init__()

        self.model = model
        self.optim = optim

    def forward(self, batch: Batch, use: str = "obs", **kwargs: Any) -> torch.Tensor:  # (B,)
        obs = to_torch(batch[use])
        critic_value = self.model(obs)
        assert critic_value.shape == torch.Size([len(batch)])
        return critic_value

    def train_step(self, loss: torch.Tensor) -> None:
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
