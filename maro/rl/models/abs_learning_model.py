# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod

import torch.nn as nn


class AbsLearningModel(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, inputs):
        """Feedforward computation"""
        return NotImplemented

    @abstractmethod
    def step(self, *losses):
        """Use losses to back-propagate gradients and apply the gradients to the underlying parameters."""
        return NotImplemented
