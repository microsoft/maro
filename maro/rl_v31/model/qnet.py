# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from abc import ABCMeta, abstractmethod
from typing import Any

import torch
from torch.optim import Optimizer

from maro.rl_v31.model.base import BaseNet


class QNet(BaseNet, metaclass=ABCMeta):
    @abstractmethod
    def q_values(self, obs: Any, act: torch.Tensor) -> torch.Tensor:  # (B,)
        raise NotImplementedError


class ContinuousQNet(QNet, metaclass=ABCMeta):
    pass


class DiscreteQNet(QNet):
    def q_values(self, obs: Any, act: torch.Tensor) -> torch.Tensor:  # (B,)
        q = self.q_values_for_all(obs)  # (B, action_num)
        return q.gather(1, act.unsqueeze(1).long()).squeeze(-1)  # (B, action_num) & (B,) => (B,)

    @abstractmethod
    def q_values_for_all(self, obs: Any) -> torch.Tensor:  # (B, action_num)
        raise NotImplementedError


class QCritic(BaseNet, metaclass=ABCMeta):
    def __init__(self, model: QNet, optim: Optimizer) -> None:
        super().__init__()

        self.qnet = model
        self.optim = optim

    def forward(self, obs: Any, act: torch.Tensor, **kwargs: Any) -> torch.Tensor:  # (B,)
        return self.qnet.q_values(obs, act)

    def train_step(self, loss: torch.Tensor) -> None:
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
