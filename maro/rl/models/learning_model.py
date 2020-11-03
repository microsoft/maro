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
        self._representation_stack = nn.Sequential(*blocks)
        self._net = {key: nn.Sequential(self._representation_stack, task_head) for key, task_head in heads.items()}

    def forward(self, inputs, head_key=None):
        """Feedforward computations for the given head(s).

        Args:
            inputs: Inputs to the model.
            head_key: The key(s) to the head(s) from which the output is required. If this is None, the results from
                all heads will be returned in the form of a dictionary. If this is a list, the results will be the
                outputs from the heads contained in head_key in the form of a dictionary. If this is a single key,
                the result will be the output from the corresponding head.

        Returns:
            Outputs from the required head(s).
        """
        if not self._net:
            return self._representation_stack(inputs)

        if head_key is None:
            return {key: net(inputs) for key, net in self._net.items()}

        if isinstance(head_key, list):
            return {key: self._net[key](inputs) for key in head_key}
        else:
            return self._net[head_key](inputs)
