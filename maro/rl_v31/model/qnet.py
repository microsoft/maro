# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from abc import ABCMeta, abstractmethod
from typing import Any

import torch

from maro.rl_v31.model.base import BaseNet


class QNet(BaseNet, metaclass=ABCMeta):
    @abstractmethod
    def q_values(self, obs: Any, act: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ContinuousQNet(QNet, metaclass=ABCMeta):
    pass


class DiscreteQNet(QNet):
    def q_values(self, obs: Any, act: torch.Tensor) -> torch.Tensor:
        q = self.q_values_for_all(obs)  # [B, action_num]
        return q.gather(1, act.long()).reshape(-1)  # [B, action_num] & [B, 1] => [B]

    @abstractmethod
    def q_values_for_all(self, obs: Any) -> torch.Tensor:
        raise NotImplementedError
