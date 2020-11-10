# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn

from maro.utils import clone
from maro.utils.exception.rl_toolkit_exception import MissingOptimizerError
from .abs_learning_model import AbsLearningModel


class SingleTaskLearningModel(AbsLearningModel):
    """NN model that consists of a sequence of chainable blocks.

    Args:
        block_list (list): List of blocks that compose the model. They must be chainable, i.e., the output dimension
            of a block must match the input dimension of its successor.
        optimizer_opt (tuple): Optimizer option for the model. Default to None.
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


class MultiTaskLearningModel(AbsLearningModel):
    """NN model that consists of multiple task heads and an optional shared stack.

    Args:
        task_block_dict (dict): Dictionary of network blocks that perform designated tasks.
        task_optimizer_opt_dict (dict): Dictionary of optimizer options for each task block. An optimizer option
            is specified in the form of a tuple: (optimizer class, optimizer parameters). Defaults to None.
        shared_block_list (list): List of blocks that compose the bottom stack of the model shared by all tasks.
            The shared blocks must be chainable, i.e., the output dimension of a block must match the input dimension
            of its successor. Defaults to None.
        shared_optimizer_opt (tuple): Optimizer option for the shared part of the model. Default to None.
    """
    def __init__(
        self,
        task_block_dict: dict,
        task_optimizer_opt_dict: dict = None,
        shared_block_list: list = None,
        shared_optimizer_opt: tuple = None
    ):
        super().__init__()
        self._has_shared_layers = shared_block_list is not None
        self._has_trainable_shared_layers = self._has_shared_layers and shared_optimizer_opt is not None
        self._has_trainable_heads = task_optimizer_opt_dict is not None

        # shared stack
        if self._has_shared_layers:
            self._shared_stack = nn.Sequential(*shared_block_list)
            if self._has_trainable_shared_layers:
                self._shared_optimizer = shared_optimizer_opt[0](
                    self._shared_stack.parameters(), **shared_optimizer_opt[1]
                )
            else:
                for param in self._shared_stack.parameters():
                    param.requires_grad = False

        # heads
        self._tasks = list(task_block_dict.keys())
        self._net = nn.ModuleDict({
            key: nn.Sequential(self._shared_stack, head) if self._has_shared_layers else head
            for key, head in task_block_dict.items()
        })

        if self._has_trainable_heads:
            self._head_optimizer_dict = {
                key: task_optimizer_opt_dict[key][0](head.parameters(), **task_optimizer_opt_dict[key][1])
                for key, head in task_block_dict.items()
            }
        else:
            for key, head in task_block_dict.items():
                for param in head.parameters():
                    param.requires_grad = False

    def __getstate__(self):
        dic = self.__dict__.copy()
        if "_shared_optimizer" in dic:
            del dic["_shared_optimizer"]
        if "_head_optimizer_dict" in dic:
            del dic["_head_optimizer_dict"]
        return dic

    def __setstate__(self, dic: dict):
        self.__dict__ = dic

    def __getitem__(self, task):
        return self._net[task]

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
    def tasks(self) -> [str]:
        return self._tasks

    def forward(self, inputs, task=None):
        """Feedforward computations for the given head(s).

        Args:
            inputs: Inputs to the model.
            task: The task for which the network output is required. If this is None, the results from all task
                heads will be returned in the form of a dictionary. If this is a list, the results will be the
                outputs from the heads contained in task in the form of a dictionary. If this is a single key,
                the result will be the output from the corresponding head.

        Returns:
            Outputs from the required head(s).
        """
        if task is None:
            return {key: self._net[key](inputs) for key in self._tasks}

        if isinstance(task, list):
            return {k: self._net[k](inputs) for k in task}
        else:
            return self._net[task](inputs)

    def step(self, *losses):
        """Use the losses to back-propagate gradients and apply them to the underlying parameters."""
        if not self._has_trainable_shared_layers and not self._has_trainable_heads:
            raise MissingOptimizerError("No optimizer registered to the model")

        # Zero all gradients
        if self._has_trainable_shared_layers:
            self._shared_optimizer.zero_grad()
        if self._has_trainable_heads:
            for optim in self._head_optimizer_dict.values():
                optim.zero_grad()

        # Accumulate gradients from all losses
        for loss in losses:
            loss.backward()

        # Apply gradients
        if self._has_trainable_shared_layers:
            self._shared_optimizer.step()
        if self._has_trainable_heads:
            for optim in self._head_optimizer_dict.values():
                optim.step()

    def copy(self):
        return clone(self)
