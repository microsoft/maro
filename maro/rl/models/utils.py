# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.utils.exception.rl_toolkit_exception import UnchainableModuleError


def check_chainability(init_func):
    def decorator(self, *task_modules, shared_module=None):
        if shared_module is not None:
            for task_module in task_modules:
                if shared_module.output_dim != task_module.input_dim:
                    raise UnchainableModuleError(
                        f"Expected input dimension {shared_module.output_dim} for {task_module.name}, "
                        f"got {task_module.input_dim}")
        init_func(self, *task_modules, shared_module=shared_module)

    return decorator
