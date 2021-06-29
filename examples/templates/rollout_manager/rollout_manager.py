# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

from maro.rl import LocalRolloutManager, MultiNodeRolloutManager, MultiProcessRolloutManager

example_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # example directory
sys.path.insert(0, example_dir)
from general import config, get_agent_wrapper_func_index, get_env_wrapper_func_index, log_dir


def get_rollout_manager():
    rollout_mode = config["rollout"]["mode"]
    if rollout_mode == "single-process":
        return LocalRolloutManager(
            get_env_wrapper_func_index[config["scenario"]](),
            num_steps=config["num_steps"],
            log_dir=log_dir
        )
    if rollout_mode == "multi-process":
        return MultiProcessRolloutManager(
            config["rollout"]["num_workers"],
            get_env_wrapper_func_index[config["scenario"]],
            get_agent_wrapper_func_index[config["scenario"]],
            num_steps=config["num_steps"],
            log_dir=log_dir,
        )
    if rollout_mode == "multi-node":
        return MultiNodeRolloutManager(
            config["rollout"]["group"],
            config["rollout"]["num_workers"],
            proxy_kwargs={"redis_address": (config["redis"]["host"], config["redis"]["port"])}
        )

    raise ValueError(
        f"Unsupported roll-out mode: {rollout_mode}. Supported modes: single-process, multi-process, multi-node"
    )
