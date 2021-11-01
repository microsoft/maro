# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import importlib
import os
import sys
from typing import List, Union


def from_env(var_name, required=True, default=None):
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


def get_eval_schedule(sch: Union[int, List[int]], num_episodes: int):
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


def get_scenario_module(scenario_path):
    scenario_dir = os.path.normpath(scenario_path)
    sys.path.insert(0, os.path.dirname(scenario_dir))
    return importlib.import_module(os.path.basename(scenario_dir))


def get_log_dir(dir: str, job_name: str):
    return os.path.join(dir, job_name)


def get_checkpoint_dir(dir: str = None):
    if dir:
        os.makedirs(dir, exist_ok=True)
    return dir


def get_load_policy_dir(dir: str):
    return from_env("LOADDIR", required=False, default=None)
