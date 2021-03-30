from typing import Dict, List, Union

import torch
import torch.nn as nn

from maro.rl import OptimOption, AbsLearningModel


class VMMultiHeadModel(AbsLearningModel):
    """A compound network structure that consists of multiple task heads and an optional shared stack.

    Args:
        component (Union[nn.Module, Dict[str, nn.Module]]): Network component(s) comprising the model.
            All components must have the same input dimension except the one designated as the shared
            component by ``shared_component_name``.
        optim_option (Union[OptimOption, Dict[str, OptimOption]]): Optimizer option for
            the components. Defaults to None.
        shared_component_name (str): Name of the network component to be designated as the shared component at the
            bottom of the architecture. Must be None or a key in ``component``. If only a single component
            is present, this is ignored. Defaults to None.
    """
    def __init__(
        self,
        component: Union[nn.Module, Dict[str, nn.Module]],
        optim_option: Union[OptimOption, Dict[str, OptimOption]] = None,
        shared_component_name: str = None
    ):
        super().__init__(component, optim_option=optim_option)
        if isinstance(component, dict):
            if shared_component_name is not None:
                assert (shared_component_name in component), (
                    f"shared_component_name must be one of {list(component.keys())}, got {shared_component_name}"
                )
            self._task_names = [name for name in component if name != shared_component_name]
        else:
            self._task_names = None
        self._shared_component_name = shared_component_name

    @property
    def task_names(self):
        return self._task_names

    @property
    def learning_rate(self):
        return self._optimizer.param_groups[0]['lr']

    def _forward(self, inputs, task_name: str = None):
        if not isinstance(self._component, nn.ModuleDict):
            return self._component(inputs)

        if self._shared_component_name is not None:
            inputs = self._component[self._shared_component_name](inputs)  # features

        if task_name is None:
            return {name: self._component[name](inputs) for name in self._task_names}

        if isinstance(task_name, list):
            return {name: self._component[name](inputs) for name in task_name}
        else:
            return self._component[task_name](inputs)

    def forward(self, inputs, task_name: Union[str, List[str]] = None, is_training: bool = True):
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