# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn


class SingleHeadLearningModel(nn.Module):
    """NN model that consists of shared blocks and multiple task heads.

    The shared blocks must be chainable, i.e., the output dimension of a block must match the input dimension of
    its successor.
    """
    def __init__(self, block_list: list, optimizer_cls, optimizer_params: dict):
        super().__init__()
        self._net = nn.Sequential(*block_list)
        self._optimizer = optimizer_cls(self._net.parameters(), **optimizer_params)

    def forward(self, inputs):
        """Feedforward computation.

        Args:
            inputs: Inputs to the model.

        Returns:
            Outputs from the model.
        """
        return self._net(inputs)

    def step(self, loss: torch.tensor):
        """Feedforward computation.

        Args:
            loss: loss tensor

        Returns:
            Outputs from the model.
        """
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()


class MultiHeadLearningModel(nn.Module):
    """NN model that consists of shared blocks and multiple task heads.

    The shared blocks must be chainable, i.e., the output dimension of a block must match the input dimension of
    its successor. Heads must be provided in the form of keyword arguments. If at least one head is provided, the
    output of the model will be a dictionary with the names of the heads as keys and the corresponding head outputs
    as values. Otherwise, the output will be the output of the last block.
    """
    def __init__(self, shared_block_list: list, task_head_block_dict: dict):
        super().__init__()
        self._task_head_keys = list(task_head_block_dict.keys())
        shared_stack = nn.Sequential(*shared_block_list)
        for key, head in task_head_block_dict.items():
            setattr(self, key, nn.Sequential(shared_stack, head))

    def forward(self, inputs, key=None):
        """Feedforward computations for the given head(s).

        Args:
            inputs: Inputs to the model.
            key: The key(s) to the head(s) from which the output is required. If this is None, the results from
                all heads will be returned in the form of a dictionary. If this is a list, the results will be the
                outputs from the heads contained in head_key in the form of a dictionary. If this is a single key,
                the result will be the output from the corresponding head.

        Returns:
            Outputs from the required head(s).
        """
        if key is None:
            return {key: getattr(self, key)(inputs) for key in self._task_head_keys}

        if isinstance(key, list):
            return {k: getattr(self, k)(inputs) for k in key}
        else:
            return getattr(self, key)(inputs)
