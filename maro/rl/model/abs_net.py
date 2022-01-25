# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any, Dict

import torch.nn


class AbsNet(torch.nn.Module, metaclass=ABCMeta):
    """
    Base class for all Torch net classes. It has the interfaces for both training and inference.
    """
    def __init__(self) -> None:
        super(AbsNet, self).__init__()

    @abstractmethod
    def step(self, loss: torch.Tensor) -> None:
        """
        Run a training step to update its own parameters according to the given loss.

        Args:
            loss (torch.tensor): Loss used to update the model.
        """
        raise NotImplementedError

    @abstractmethod
    def get_gradients(self, loss: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get the gradients with respect to all parameters according to the given loss.

        Args:
            loss (torch.tensor): Loss used to update the model.

        Returns:
            A dict that contains gradients for all parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_gradients(self, grad: dict) -> None:
        raise NotImplementedError

    def _forward_unimplemented(self, *input: Any) -> None:  # TODO
        pass

    @abstractmethod
    def get_net_state(self) -> object:
        """
        Get the net's state.
        """
        raise NotImplementedError

    @abstractmethod
    def set_net_state(self, net_state: object) -> None:
        """
        Set the net's state.
        """
        raise NotImplementedError

    def soft_update(self, other_model: AbsNet, tau: float) -> None:
        """
        Soft update the net's parameters according to another net.

        Args:
            other_model (AbsNet): The source net. Must has same type with the current net.
            tau (float): Soft update coefficient.
        """
        assert self.__class__ == other_model.__class__, \
            f"Soft update can only be done between same classes. Current model type: {self.__class__}, " \
            f"other model type: {other_model.__class__}"

        for params, other_params in zip(self.parameters(), other_model.parameters()):
            params.data = (1 - tau) * params.data + tau * other_params.data

    @abstractmethod
    def freeze(self) -> None:
        """
        (Partially) freeze the current model. The users should write their own strategy to determine the list of
        parameters to freeze.
        """
        raise NotImplementedError

    @abstractmethod
    def unfreeze(self) -> None:
        """
        (Partially) unfreeze the current model. The users should write their own strategy to determine the list of
        parameters to unfreeze.
        """
        raise NotImplementedError

    def freeze_all_parameters(self) -> None:
        """
        Freeze all parameters.
        """
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_all_parameters(self) -> None:
        """
        Unfreeze all parameters.
        """
        for p in self.parameters():
            p.requires_grad = True
