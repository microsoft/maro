# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os.path import dirname, realpath

from maro.rl import LocalPolicyManager, MultiNodePolicyManager, MultiProcessPolicyManager

template_dir = dirname(dirname(realpath(__file__)))  # template directory
if template_dir not in sys.path:
    sys.path.insert(0, template_dir)
from general import config, create_train_policy_func, log_dir

def get_policy_manager():
    train_mode = config["policy_manager"]["train_mode"]
    num_trainers = config["policy_manager"]["num_trainers"]
    policy_dict = {name: func() for name, func in create_train_policy_func.items()}
    if train_mode == "single-process":
        return LocalPolicyManager(policy_dict, log_dir=log_dir)
    if train_mode == "multi-process":
        return MultiProcessPolicyManager(
            policy_dict,
            num_trainers,
            create_train_policy_func,
            log_dir=log_dir
        )
    if train_mode == "multi-node":
        return MultiNodePolicyManager(
            policy_dict,
            config["policy_manager"]["train_group"],
            num_trainers,
            proxy_kwargs={"redis_address": (config["redis"]["host"], config["redis"]["port"])},
            log_dir=log_dir
        )

    raise ValueError(
        f"Unsupported policy training mode: {train_mode}. Supported modes: single-process, multi-process, multi-node"
    )
