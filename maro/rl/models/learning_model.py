# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn


class LearningModel(nn.Module):
    """A general NN model that consists of multiple building blocks.

    The building blocks must be chainable, i.e., the output dimension of one block must match the input dimension of
    its successors.
    """
    def __init__(self, *blocks):
        super().__init__()
        self._net = nn.Sequential(*blocks)

    def forward(self, inputs):
        return self._net(inputs)
