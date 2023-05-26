# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from abc import ABCMeta, abstractmethod
from typing import Any, Optional

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
    def q_values(self, obs: Any, act: Optional[torch.Tensor] = None) -> torch.Tensor:  # (B,) or (B, action_num)
        if act is not None:
            q = self.q_values(obs)  # (B, action_num)
            return q.gather(1, act.unsqueeze(1)).squeeze(-1).float()  # (B, action_num) & (B,) => (B,)
        else:
            return self.q_values_for_all(obs).float()

    @abstractmethod
    def q_values_for_all(self, obs: Any) -> torch.Tensor:  # (B, action_num)
        raise NotImplementedError


class QCritic(BaseNet, metaclass=ABCMeta):
    def __init__(self, model: QNet, optim: Optimizer) -> None:
        super().__init__()

        self.model = model
        self.optim = optim

    def forward(self, obs: Any, act: torch.Tensor, **kwargs: Any) -> torch.Tensor:  # (B,)
        critic_value = self.model.q_values(obs, act)
        assert critic_value.shape == torch.Size([len(obs)])
        return critic_value

    def train_step(self, loss: torch.Tensor) -> None:
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
