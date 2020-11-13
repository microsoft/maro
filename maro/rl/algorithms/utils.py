# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum
from functools import wraps

import numpy as np
import torch

from maro.utils.exception.rl_toolkit_exception import UnrecognizedTaskError


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def to_device(init_func):
    @wraps(init_func)
    def wrapper(self, model, config):
        init_func(self, model.to(device), config)

    return wrapper


def preprocess(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        converted_args = [
            torch.from_numpy(arg).to(device) if isinstance(arg, np.ndarray) else arg for arg in args
        ]
        converted_kwargs = {
            kw: torch.from_numpy(arg).to(device) if isinstance(arg, np.ndarray) else arg
            for kw, arg in kwargs.items()
        }
        return func(*converted_args, **converted_kwargs)

    return wrapper
