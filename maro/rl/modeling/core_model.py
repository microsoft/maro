# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod

import torch
import torch.nn as nn


class AbsCoreModel(nn.Module):
    """General model abstraction for 

    Args:
        device (str): Identifier for the torch device. The model instance will be moved to the specified
            device. If it is None, the device will be set to "cpu" if cuda is unavailable and "cuda" otherwise.
            Defaults to None.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def step(self, loss: torch.tensor):
        """Use the loss to back-propagate gradients and apply them to the underlying parameters.

        Args:
            loss: Result of a computation graph that involves the underlying parameters.
        """
        raise NotImplementedError

    def get_gradients(self, loss: torch.tensor):
        pass 

    def apply_gradients(self, grad: dict):
        pass
