# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import namedtuple
from typing import Dict, Union

import torch.nn as nn

from maro.utils.exception.rl_toolkit_exception import MissingOptimizerError

from .abs_learning_model import AbsLearningModel


OptimizerOptions = namedtuple("OptimizerOptions", ["cls", "params"])


class LearningModule(nn.Module):
    """NN model that consists of a sequence of chainable blocks.

    Args:
        block_list (list): List of blocks that compose the model. They must be chainable, i.e., the output dimension
            of a block must match the input dimension of its successor.
        optimizer_options (OptimizerOptions): A namedtuple of (optimizer_class, optimizer_parameters).
    """
    def __init__(self, name: str, block_list: list, optimizer_options: OptimizerOptions = None):
        super().__init__()
        self._name = name
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
        dic["is_trainable"] = False
        return dic

    def __setstate__(self, dic: dict):
        self.__dict__ = dic

    @property
    def name(self):
        return self._name

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

    def zero_grad(self):
        if not self._is_trainable:
            raise MissingOptimizerError("No optimizer registered to the model")
        self._optimizer.zero_grad()

    def step(self):
        self._optimizer.step()


class MultiTaskLearningModel(AbsLearningModel):
    """NN model that consists of multiple task heads and an optional shared stack.

    Args:
        task_modules (LearningModule): LearningModule instances, each of which performs a designated task.
        shared_module (LearningModule): Network module that forms that shared part of the model. Defaults to
            None.
    """
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
        self._task_modules = task_modules
        self._net = nn.ModuleDict({
            task_module.name: nn.Sequential(self._shared_module, task_module) if self._shared_module else task_module
            for task_module in self._task_modules
        })

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
    def task_names(self) -> [str]:
        return self._task_names

    @property
    def is_trainable(self) -> bool:
        return any(task_module.is_trainable for task_module in self._task_modules) or self._shared_module.is_trainable

    def forward(self, inputs, task_name=None):
        """Feedforward computations for the given head(s).

        Args:
            inputs: Inputs to the model.
            task_name: The name of the task for which the network output is required. If this is None, the results from
                all task heads will be returned in the form of a dictionary. If this is a list, the results will be the
                outputs from the heads contained in task in the form of a dictionary. If this is a single key, the
                result will be the output from the corresponding head.

        Returns:
            Outputs from the required head(s).
        """
        if task_name is None:
            return {key: self._net[key](inputs) for key in self._task_names}

        if isinstance(task_name, list):
            return {k: self._net[k](inputs) for k in task_name}
        else:
            return self._net[task_name](inputs)

    def learn(self, loss):
        """Use the loss to back-propagate gradients and apply them to the underlying parameters."""
        for task_module in self._task_modules:
            task_module.zero_grad()
        if self._shared_module is not None:
            self._shared_module.zero_grad()

        # Obtain gradients through back-propagation
        loss.backward()

        # Apply gradients
        for task_module in self._task_modules:
            task_module.step()
        if self._shared_module is not None:
            self._shared_module.step()
