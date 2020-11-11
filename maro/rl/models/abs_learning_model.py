# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod

import torch
import torch.nn as nn

from maro.utils import clone


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

    def copy(self):
        return clone(self)

    def load(self, state_dict):
        self.load_state_dict(state_dict)

    def dump(self):
        return self.state_dict()

    def load_from_file(self, path: str):
        self.load_state_dict(torch.load(path))

    def dump_to_file(self, path: str):
        torch.save(self.state_dict(), path)
