# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from collections import namedtuple
from typing import Dict, Union

import torch
import torch.nn as nn

from maro.utils import clone
from maro.utils.exception.rl_toolkit_exception import MissingOptimizer, NNStackDimensionError

from .abs_block import AbsBlock

OptimizerOptions = namedtuple("OptimizerOptions", ["cls", "params"])


class NNStack(nn.Module):
    """An NN stack that consists of a sequence of chainable blocks.

    Args:
        name (str): Name of the stack.
        blocks (AbsBlock): Blocks that comprise the model. They must be chainable, i.e., the output dimension
            of a block must match the input dimension of its successor.
    """
    def __init__(self, name: str, *blocks: [AbsBlock]):
        super().__init__()
        self._name = name
        self._net = nn.Sequential(*blocks)

    @property
    def name(self):
        return self._name

    def forward(self, inputs):
        """Feedforward computation.

        Args:
            inputs: Inputs to the model.

        Returns:
            Outputs from the model.
        """
        return self._net(inputs)


class AbsLearningModel(nn.Module):
    """NN model that consists of NN stacks.

    Args:
        stacks (NNStack): NNStack instances.
        optimizer_options (Union[OptimizerOptions, Dict[str, OptimizerOptions]]): Optimizer options for
            the internal stacks. If none, no optimizer will be created for the model and the model will not
            be trainable. If it is a single OptimizerOptions instance, an optimizer will be created to jointly
            optimize all parameters of the model. If it is a dictionary, for each `(key, value)` pair, an optimizer
            specified by `value` will be created for the internal stack named `key`. Defaults to None.
    """
    def __init__(
        self,
        *stacks: NNStack,
        optimizer_options: Union[OptimizerOptions, Dict[str, OptimizerOptions]] = None
    ):
        super().__init__()
        self._component = nn.ModuleDict({stack.name: stack for stack in stacks})
        self._is_trainable = optimizer_options is not None
        if self._is_trainable:
            if isinstance(optimizer_options, OptimizerOptions):
                self._optimizer = optimizer_options.cls(self.parameters(), **optimizer_options.params)
            else:
                self._optimizer = {
                    stack_name: opt.cls(self._component[stack_name].parameters(), **opt.params)
                    for stack_name, opt in optimizer_options.items()
                }
        else:
            self.eval()
            for param in self.parameters():
                param.requires_grad = False

    def __getstate__(self):
        dic = self.__dict__.copy()
        if "_optimizer" in dic:
            del dic["_optimizer"]
        dic["_is_trainable"] = False
        return dic

    def __setstate__(self, dic: dict):
        self.__dict__ = dic

    @property
    def is_trainable(self) -> bool:
        return self._is_trainable

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def learn(self, loss):
        """Use the loss to back-propagate gradients and apply them to the underlying parameters."""
        if not self._is_trainable:
            raise MissingOptimizer("No optimizer registered to the model")
        if isinstance(self._optimizer, dict):
            for optimizer in self._optimizer.values():
                optimizer.zero_grad()
        else:
            self._optimizer.zero_grad()

        # Obtain gradients through back-propagation
        loss.backward()

        # Apply gradients
        if isinstance(self._optimizer, dict):
            for optimizer in self._optimizer.values():
                optimizer.step()
        else:
            self._optimizer.step()

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


class SimpleMultiHeadedModel(AbsLearningModel):
    """NN model that consists of multiple task heads and an optional shared stack.

    Args:
        task_stacks (NNStack): NNStack instances, each of which performs a designated task.
        shared_stack (NNStack): Network module that forms that shared part of the model. Defaults to None.
        optimizer_options (Union[OptimizerOptions, Dict[str, OptimizerOptions]]): Optimizer options for
            the internal stacks. If none, no optimizer will be created for the model and the model will not
            be trainable. If it is a single OptimizerOptions instance, an optimizer will be created to jointly
            optimize all parameters of the model. If it is a dictionary, for each `(key, value)` pair, an optimizer
            specified by `value` will be created for the internal stack named `key`. Defaults to None.
    """
    def __init__(
        self,
        *task_stacks: NNStack,
        shared_stack: NNStack = None,
        optimizer_options: Union[OptimizerOptions, Dict[str, OptimizerOptions]] = None
    ):
        self.validate_dims(*task_stacks, shared_stack=shared_stack)
        self._task_names = [stack.name for stack in task_stacks]
        stacks = task_stacks + (shared_stack,) if shared_stack else task_stacks
        super().__init__(*stacks, optimizer_options=optimizer_options)
        self._shared_stack = shared_stack

    @property
    def task_names(self) -> [str]:
        return self._task_names

    @property
    def shared_stack(self):
        return self._shared_stack

    def _forward(self, inputs, task_name: str = None):
        if self._shared_stack:
            inputs = self._shared_stack(inputs)  # features

        if len(self._component) == 1:
            return list(self._component.values())[0](inputs)

        if task_name is None:
            return {name: self._component[name](inputs) for name in self._task_names}

        if isinstance(task_name, list):
            return {name: self._component[name](inputs) for name in task_name}
        else:
            return self._component[task_name](inputs)

    def forward(self, inputs, task_name: str = None, is_training: bool = True):
        """Feedforward computations for the given head(s).

        Args:
            inputs: Inputs to the model.
            task_name (str): The name of the task for which the network output is required. If the model contains only
                one task module, the task_name is ignored and the output of that module will be returned. If the model
                contains multiple task modules, then 1) if task_name is None, the output from all task modules will be
                returned in the form of a dictionary; 2) if task_name is a list, the outputs from the task modules
                specified in the list will be returned in the form of a dictionary; 3) if this is a single string,
                the output from the corresponding task module will be returned.
            is_training (bool): If true, all torch submodules will be set to training mode, and auto-differentiation
                will be turned on. Defaults to True.

        Returns:
            Outputs from the required head(s).
        """
        self.train(mode=is_training)
        if is_training:
            return self._forward(inputs, task_name)

        with torch.no_grad():
            return self._forward(inputs, task_name)

    def soft_update(self, other_model: nn.Module, tau: float):
        for params, other_params in zip(self.parameters(), other_model.parameters()):
            params.data = (1 - tau) * params.data + tau * other_params.data

    @staticmethod
    def validate_dims(*task_stacks, shared_stack=None):
        if shared_stack:
            expected_dim = shared_stack.output_dim
            for task_stack in task_stacks:
                if task_stack.input_dim != expected_dim:
                    raise NNStackDimensionError(
                        f"Expected input dimension {expected_dim} for task module: {task_stack.name}, "
                        f"got {task_stack.input_dim}")
