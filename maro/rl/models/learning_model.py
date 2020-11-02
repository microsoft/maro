# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch.nn as nn


class LearningModel(nn.Module):
    """NN model that consists of multiple shared blocks and multiple heads.

    The shared blocks must be chainable, i.e., the output dimension of a block must match the input dimension of
    its successor. Heads must be provided in the form of keyword arguments. If at least one head is provided, the
    output of the model will be a dictionary with the names of the heads as keys and the corresponding head outputs
    as values. Otherwise, the output will be the output of the last block.
    """
    def __init__(self, *blocks, **heads):
        super().__init__()
        self._shared = nn.Sequential(*blocks)
        self._heads = heads

    def forward(self, inputs):
        if not self._heads:
            return self._shared(inputs)
        else:
            features = self._shared(inputs)
            return {name: layers(features) for name, layers in self._heads.items()}
