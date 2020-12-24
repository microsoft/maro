# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.utils.exception.rl_toolkit_exception import LearningModuleDimensionError


def validate_dims(init_func):
    def decorator(self, *task_modules, shared_module=None):
        expected_dim = shared_module.output_dim if shared_module else task_modules[0].input_dim
        for task_module in task_modules:
            if task_module.input_dim != expected_dim:
                raise LearningModuleDimensionError(
                    f"Expected input dimension {expected_dim} for {task_module.name}, "
                    f"got {task_module.input_dim}")

        init_func(self, *task_modules, shared_module=shared_module)

    return decorator
