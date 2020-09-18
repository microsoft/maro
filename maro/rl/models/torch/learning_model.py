# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn


class IdentityLayers(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return inputs


class LearningModel(nn.Module):
    def __init__(self,
                 representation_layers: nn.Module = IdentityLayers(),
                 decision_layers: nn.Module = IdentityLayers(),
                 clip_value: float = None):
        """
        Args:
            representation_layers: an NN-based feature extractor.
            decision_layers: an NN model that takes the output of the representation layers as input and
                             outputs values of interest in RL (e.g., state & action values).
            clip_value: threshold used to clip gradients.
        """
        super().__init__()
        self._net = nn.Sequential(representation_layers, decision_layers)

        if clip_value is not None:
            for param in self._net.parameters():
                param.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

    def forward(self, inputs):
        return self._net(inputs)
