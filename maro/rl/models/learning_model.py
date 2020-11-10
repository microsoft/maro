# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn

from maro.utils import clone
from maro.utils.exception.rl_toolkit_exception import MissingOptimizerError
from .abs_learning_model import AbsLearningModel


class SingleHeadLearningModel(AbsLearningModel):
    """NN model that consists of shared blocks and multiple task heads.

    The shared blocks must be chainable, i.e., the output dimension of a block must match the input dimension of
    its successor.
    """
    def __init__(self, block_list: list, optimizer_opt: tuple = None):
        super().__init__()
        self._net = nn.Sequential(*block_list)
        self._is_trainable = optimizer_opt is not None
        if self._is_trainable:
            self._optimizer = optimizer_opt[0](self._net.parameters(), **optimizer_opt[1])
        else:
            for param in self._net.parameters():
                param.requires_grad = False

    def __getstate__(self):
        dic = self.__dict__.copy()
        if "_optimizer" in dic:
            del dic["_optimizer"]

        return dic

    def __setstate__(self, dic: dict):
        self.__dict__ = dic

    @property
    def is_trainable(self):
        return self._is_trainable

    def forward(self, inputs):
        """Feedforward computation.

        Args:
            inputs: Inputs to the model.

        Returns:
            Outputs from the model.
        """
        return self._net(inputs)

    def step(self, loss: torch.tensor):
        """Use the loss to back-propagate gradients and apply them to the underlying parameters."""
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def copy(self):
        return clone(self)


class MultiHeadLearningModel(AbsLearningModel):
    """NN model that consists of shared blocks and multiple task heads.

    The shared blocks must be chainable, i.e., the output dimension of a block must match the input dimension of
    its successor. Heads must be provided in the form of keyword arguments. If at least one head is provided, the
    output of the model will be a dictionary with the names of the heads as keys and the corresponding head outputs
    as values. Otherwise, the output will be the output of the last block.
    """
    def __init__(
        self,
        head_block_dict: dict,
        head_optimizer_opt_dict: dict = None,
        shared_block_list: list = None,
        shared_stack_optimizer_opt: tuple = None
    ):
        super().__init__()
        self._has_shared_layers = shared_block_list is not None
        self._has_trainable_shared_layers = self._has_shared_layers and shared_stack_optimizer_opt is not None
        self._has_trainable_heads = head_optimizer_opt_dict is not None

        # shared stack
        if self._has_shared_layers:
            self._shared_stack = nn.Sequential(*shared_block_list)
            if self._has_trainable_shared_layers:
                self._shared_stack_optimizer = shared_stack_optimizer_opt[0](
                    self._shared_stack.parameters(), **shared_stack_optimizer_opt[1]
                )
            else:
                for param in self._shared_stack.parameters():
                    param.requires_grad = False

        # heads
        self._head_keys = list(head_block_dict.keys())
        self._net = nn.ModuleDict({
            key: nn.Sequential(self._shared_stack, head) if self._has_shared_layers else head
            for key, head in head_block_dict.items()
        })

        if self._has_trainable_heads:
            self._head_optimizer_dict = {
                key: head_optimizer_opt_dict[key][0](head.parameters(), **head_optimizer_opt_dict[key][1])
                for key, head in head_block_dict.items()
            }
        else:
            for key, head in head_block_dict.items():
                for param in head.parameters():
                    param.requires_grad = False

    def __getstate__(self):
        dic = self.__dict__.copy()
        if "_shared_stack_optimizer" in dic:
            del dic["_shared_stack_optimizer"]
        if "_head_optimizer_dict" in dic:
            del dic["_head_optimizer_dict"]
        return dic

    def __setstate__(self, dic: dict):
        self.__dict__ = dic

    @property
    def has_shared_layers(self):
        return self._has_shared_layers

    @property
    def has_trainable_shared_layers(self):
        return self._has_trainable_shared_layers

    @has_trainable_shared_layers.setter
    def has_trainable_shared_layers(self, value: bool):
        self._has_trainable_shared_layers = value

    @property
    def has_trainable_heads(self):
        return self._has_trainable_heads

    @has_trainable_heads.setter
    def has_trainable_heads(self, value: bool):
        self._has_trainable_heads = value

    @property
    def is_trainable(self):
        return self._has_trainable_shared_layers or self._has_trainable_heads

    @property
    def heads(self) -> [str]:
        return self._head_keys

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
            return {key: self._net[key](inputs) for key in self._head_keys}

        if isinstance(key, list):
            return {k: self._net[k](inputs) for k in key}
        else:
            return self._net[key](inputs)

    def step(self, *losses):
        """Use the losses to back-propagate gradients and apply them to the underlying parameters."""
        if not self._has_trainable_shared_layers and not self._has_trainable_heads:
            raise MissingOptimizerError("No optimizer registered to the model")

        # Zero all gradients
        if self._has_trainable_shared_layers:
            self._shared_stack_optimizer.zero_grad()
        if self._has_trainable_heads:
            for optim in self._head_optimizer_dict.values():
                optim.zero_grad()

        # Accumulate gradients from all losses
        for loss in losses:
            loss.backward()

        # Apply gradients
        if self._has_trainable_shared_layers:
            self._shared_stack_optimizer.step()
        if self._has_trainable_heads:
            for optim in self._head_optimizer_dict.values():
                optim.step()

    def copy(self):
        return clone(self)
