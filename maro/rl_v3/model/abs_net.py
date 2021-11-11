from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any, Dict

import torch.nn


class AbsNet(torch.nn.Module, metaclass=ABCMeta):
    def __init__(self) -> None:
        super(AbsNet, self).__init__()

    @abstractmethod
    def step(self, loss: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_gradients(self, loss: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def _forward_unimplemented(self, *input: Any) -> None:  # TODO
        pass

    @abstractmethod
    def get_net_state(self) -> object:
        raise NotImplementedError

    @abstractmethod
    def set_net_state(self, net_state: object) -> None:
        raise NotImplementedError

    def soft_update(self, other_model: AbsNet, tau: float) -> None:
        assert self.__class__ == other_model.__class__, \
            f"Soft update can only be done between same classes. Current model type: {self.__class__}, " \
            f"other model type: {other_model.__class__}"

        for params, other_params in zip(self.parameters(), other_model.parameters()):
            params.data = (1 - tau) * params.data + tau * other_params.data

    @abstractmethod
    def freeze(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def unfreeze(self) -> None:
        raise NotImplementedError

    def freeze_all_parameters(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_all_parameters(self) -> None:
        for p in self.parameters():
            p.requires_grad = True
