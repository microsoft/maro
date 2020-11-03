# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch.nn as nn


class DuelingActionValueModel(nn.Module):
    def __init__(self, value_head: nn.Module, advantage_head: nn.Module, shared_layers:nn.Module = nn.Identity(),
                 ):
        super().__init__()
        self._value_head = value_head
        self._advantage_head = advantage_head
        self._shared_layers = shared_layers
        if self._advantage_mode not in {"mean", "max"}:
            raise ValueError("Advantage mode must be 'mean' or 'max'")
        self._advantage_mode = advantage_mode

    def forward(self, inputs):
        features = self._shared_layers(inputs)
        state_values = self._value_head(features)
        advantages = self._advantage_head(features)
        # use mean or max correction to address the identifiability issue
        corrections = advantages.mean(1) if self._advantage_mode == "mean" else advantages.max(1)[0]
        q_values = state_values + advantages - corrections.unsqueeze(1)
        return q_values
