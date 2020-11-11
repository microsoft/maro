# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum

from maro.rl.models.learning_model import MultiTaskLearningModel
from maro.utils.exception.rl_toolkit_exception import UnrecognizedTaskError


def validate_tasks(task_enum: Enum):
    def decorator(init_func):
        def wrapper(core_model, config):
            recognized_tasks = set(member.value for member in task_enum)
            model_tasks = set(core_model.tasks)
            if isinstance(core_model, MultiTaskLearningModel) and model_tasks != recognized_tasks:
                raise UnrecognizedTaskError(f"Expected task names {recognized_tasks}, got {model_tasks}")

            init_func(core_model, config)

        return wrapper

    return decorator
