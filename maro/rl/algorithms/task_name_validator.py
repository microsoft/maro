# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum
from functools import wraps

from maro.utils.exception.rl_toolkit_exception import UnrecognizedTaskError


def validate_task_names(task_enum: Enum):
    def decorator(init_func):
        @wraps(init_func)
        def wrapper(self, core_model, config):
            recognized_task_names = set(member.value for member in task_enum)
            model_task_names = set(core_model.task_names)
            if len(model_task_names) > 1 and model_task_names != recognized_task_names:
                raise UnrecognizedTaskError(f"Expected task names {recognized_task_names}, got {model_task_names}")

            init_func(self, core_model, config)

        return wrapper

    return decorator
