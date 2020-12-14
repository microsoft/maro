# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import namedtuple

import torch
import torch.nn as nn

from maro.utils import clone
from maro.utils.exception.rl_toolkit_exception import MissingOptimizerError

from .abs_block import AbsBlock
from .utils import check_chainability

OptimizerOptions = namedtuple("OptimizerOptions", ["cls", "params"])


class LearningModule(nn.Module):
    """NN model that consists of a sequence of chainable blocks.

    Args:
        block_list (list): List of blocks that compose the model. They must be chainable, i.e., the output dimension
            of a block must match the input dimension of its successor.
        optimizer_options (OptimizerOptions): A namedtuple of (optimizer_class, optimizer_parameters).
    """
    def __init__(self, name: str, block_list: [AbsBlock], optimizer_options: OptimizerOptions = None):
        super().__init__()
        self._name = name
        self._input_dim = block_list[0].input_dim
        self._output_dim = block_list[-1].output_dim
        self._net = nn.Sequential(*block_list)
        self._is_trainable = optimizer_options is not None
        if self._is_trainable:
            self._optimizer = optimizer_options.cls(self._net.parameters(), **optimizer_options.params)
        else:
            self._net.eval()
            for param in self._net.parameters():
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
    def name(self):
        return self._name

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

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

    def zero_gradients(self):
        if not self._is_trainable:
            raise MissingOptimizerError("No optimizer registered to the model")
        self._optimizer.zero_grad()

    def step(self):
        self._optimizer.step()

    def copy(self):
        return clone(self)


class LearningModuleManager(nn.Module):
    """NN model that consists of multiple task heads and an optional shared stack.

    Args:
        task_modules (LearningModule): LearningModule instances, each of which performs a designated task.
        shared_module (LearningModule): Network module that forms that shared part of the model. Defaults to None.
    """
    @check_chainability
    def __init__(
        self,
        *task_modules: LearningModule,
        shared_module: LearningModule = None
    ):
        super().__init__()
        self._task_names = [module.name for module in task_modules]

        # shared stack
        self._shared_module = shared_module

        # task_heads
        self._task_module_dict = nn.ModuleDict({task_module.name: task_module for task_module in task_modules})

    @property
    def task_names(self) -> [str]:
        return self._task_names

    @property
    def shared_module(self):
        return self._shared_module

    @property
    def is_trainable(self) -> bool:
        return (
            any(task_module.is_trainable for task_module in self._task_module_dict.values()) or
            (self._shared_module is not None and self._shared_module.is_trainable)
        )

    def _forward(self, inputs, task_name: str = None):
        if self._shared_module:
            inputs = self._shared_module(inputs)

        if len(self._task_module_dict) == 1:
            return list(self._task_module_dict.values())[0](inputs)

        if task_name is None:
            return {name: task_module(inputs) for name, task_module in self._task_module_dict.items()}

        if isinstance(task_name, list):
            return {name: self._task_module_dict[name](inputs) for name in task_name}
        else:
            return self._task_module_dict[task_name](inputs)

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

    def learn(self, loss):
        """Use the loss to back-propagate gradients and apply them to the underlying parameters."""
        for task_module in self._task_module_dict.values():
            task_module.zero_gradients()
        if self._shared_module is not None:
            self._shared_module.zero_gradients()

        # Obtain gradients through back-propagation
        loss.backward()

        # Apply gradients
        for task_module in self._task_module_dict.values():
            task_module.step()
        if self._shared_module is not None:
            self._shared_module.step()

    def soft_update(self, other_model: nn.Module, tau: float):
        for params, other_params in zip(self.parameters(), other_model.parameters()):
            params.data = (1 - tau) * params.data + tau * other_params.data

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
