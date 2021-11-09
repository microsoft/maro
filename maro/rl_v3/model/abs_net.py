from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict

import torch.nn


class AbsNet(torch.nn.Module):
    def __init__(self) -> None:
        super(AbsNet, self).__init__()

    @abstractmethod
    def step(self, loss: torch.Tensor) -> None:
        pass

    @abstractmethod
    def get_gradients(self, loss: torch.Tensor) -> Dict[str, torch.Tensor]:
        pass

    def _forward_unimplemented(self, *input: Any) -> None:  # TODO
        pass

    @abstractmethod
    def get_net_state(self) -> object:
        pass

    @abstractmethod
    def set_net_state(self, net_state: object) -> None:
        pass

    def soft_update(self, other_model: AbsNet, tau: float) -> None:
        assert self.__class__ == other_model.__class__

        for params, other_params in zip(self.parameters(), other_model.parameters()):
            params.data = (1 - tau) * params.data + tau * other_params.data
