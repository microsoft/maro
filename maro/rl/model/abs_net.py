# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from abc import ABCMeta
from typing import Any, Dict

import torch.nn
from torch.optim import Optimizer


class AbsNet(torch.nn.Module, metaclass=ABCMeta):
    """Base class for all Torch net classes. `AbsNet` defines a set of methods that will be called by upper-level
    logic. All classes that inherit `AbsNet` should implement these methods.
    """

    def __init__(self) -> None:
        super(AbsNet, self).__init__()

    @property
    def optim(self) -> Optimizer:
        optim = getattr(self, "_optim", None)
        assert isinstance(optim, Optimizer), "Each AbsNet must have an optimizer"
        return optim

    def step(self, loss: torch.Tensor) -> None:
        """Run a training step to update the net's parameters according to the given loss.

        Args:
            loss (torch.tensor): Loss used to update the model.
        """
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def get_gradients(self, loss: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get the gradients with respect to all parameters according to the given loss.

        Args:
            loss (torch.tensor): Loss used to compute gradients.

        Returns:
            Gradients (Dict[str, torch.Tensor]): A dict that contains gradients for all parameters.
        """
        self.optim.zero_grad()
        loss.backward()
        return {name: param.grad for name, param in self.named_parameters()}

    def apply_gradients(self, grad: Dict[str, torch.Tensor]) -> None:
        """Apply gradients to the net to update all parameters.

        Args:
            grad (Dict[str, torch.Tensor]): A dict that contains gradients for all parameters.
        """
        for name, param in self.named_parameters():
            param.grad = grad[name]
        self.optim.step()

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def get_state(self) -> dict:
        """Get the net's state.

        Returns:
            state (dict): A object that contains the net's state.
        """
        return {
            "network": self.state_dict(),
            "optim": self.optim.state_dict(),
        }

    def set_state(self, net_state: dict) -> None:
        """Set the net's state.

        Args:
            net_state (dict): A dict that contains the net's state.
        """
        self.load_state_dict(net_state["network"])
        self.optim.load_state_dict(net_state["optim"])

    def soft_update(self, other_model: AbsNet, tau: float) -> None:
        """Soft update the net's parameters according to another net, i.e.,
        self.param = self.param * (1.0 - tau) + other_model.param * tau

        Args:
            other_model (AbsNet): The source net. Must has same type with the current net.
            tau (float): Soft update coefficient.
        """
        assert self.__class__ == other_model.__class__, (
            f"Soft update can only be done between same classes. Current model type: {self.__class__}, "
            f"other model type: {other_model.__class__}"
        )

        for params, other_params in zip(self.parameters(), other_model.parameters()):
            params.data = (1 - tau) * params.data + tau * other_params.data

    def freeze(self) -> None:
        """(Partially) freeze the current model. The users should write their own strategy to determine which
        parameters to freeze. Freeze all parameters is capable in most cases. You could overwrite this method
        when necessary.
        """
        self.freeze_all_parameters()

    def unfreeze(self) -> None:
        """(Partially) unfreeze the current model. The users should write their own strategy to determine which
        parameters to freeze. Unfreeze all parameters is capable in most cases. You could overwrite this method
        when necessary.
        """
        self.unfreeze_all_parameters()

    def freeze_all_parameters(self) -> None:
        """Freeze all parameters."""
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_all_parameters(self) -> None:
        """Unfreeze all parameters."""
        for p in self.parameters():
            p.requires_grad = True
