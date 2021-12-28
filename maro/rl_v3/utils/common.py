# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import importlib
import os
import sys
from types import ModuleType
from typing import List, Union

from maro.utils import Logger


def from_env(var_name: str, required: bool = True, default: object = None) -> object:
    if var_name not in os.environ:
        if required:
            raise KeyError(f"Missing environment variable: {var_name}")
        else:
            return default

    var = os.getenv(var_name)
    if var.isnumeric() or var[0] == "-" and var[1:].isnumeric():
        return int(var)

    try:
        return float(var)
    except ValueError:
        return var


def get_eval_schedule(sch: Union[int, List[int]], num_episodes: int) -> List[int]:
    """Helper function to the policy evaluation schedule.

    Args:
        sch (Union[int, List[int]]): Evaluation schedule. If it is an int, it is treated as the number of episodes
            between two adjacent evaluations. For example, if the total number of episodes is 20 and ``sch`` is 6,
            this will return [6, 12, 18] if ``final`` is False or [6, 12, 18, 20] otherwise. If it is a list, it will
            return a sorted version of the list (with the last episode appended if ``final`` is True).
        num_episodes (int): Total number of learning episodes.

    Returns:
        A list of episodes indicating when to perform policy evaluation.

    """
    if sch is None:
        schedule = []
    elif isinstance(sch, int):
        num_eval_schedule = num_episodes // sch
        schedule = [sch * i for i in range(1, num_eval_schedule + 1)]
    else:
        schedule = sorted(sch)

    return schedule


def get_module(path: str) -> ModuleType:
    path = os.path.normpath(path)
    sys.path.insert(0, os.path.dirname(path))
    return importlib.import_module(os.path.basename(path))


def get_log_path(dir: str, job_name: str) -> str:
    return os.path.join(dir, f"{job_name}.log")


def get_logger(dir: str, job_name: str, tag: str) -> Logger:
    return Logger(tag, dump_path=get_log_path(dir, job_name), dump_mode="a")


def get_checkpoint_path(dir: str = None) -> str:
    if dir:
        os.makedirs(dir, exist_ok=True)
    return dir
