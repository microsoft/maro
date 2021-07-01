# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

from maro.rl import LocalPolicyManager, MultiNodePolicyManager, MultiProcessPolicyManager

example_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # example directory
sys.path.insert(0, example_dir)
from general import config, create_train_policy_func, log_dir

def get_policy_manager():
    training_mode = config["policy_manager"]["training_mode"]
    num_trainers = config["policy_manager"]["num_trainers"]
    policy_dict = {name: func() for name, func in create_train_policy_func.items()}
    if training_mode == "single-process":
        return LocalPolicyManager(policy_dict, log_dir=log_dir)
    if training_mode == "multi-process":
        return MultiProcessPolicyManager(
            policy_dict,
            num_trainers,
            create_train_policy_func,
            log_dir=log_dir
        )
    if training_mode == "multi-node":
        return MultiNodePolicyManager(
            policy_dict,
            config["policy_manager"]["group"],
            num_trainers,
            proxy_kwargs={"redis_address": (config["redis"]["host"], config["redis"]["port"])},
            log_dir=log_dir
        )

    raise ValueError(
        f"Unsupported policy training mode: {training_mode}. "
        f"Supported modes: single-process, multi-process, multi-node"
    )
