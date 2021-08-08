# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os import getenv
from os.path import dirname, realpath

from maro.rl.policy import LocalPolicyManager, MultiNodePolicyManager, MultiProcessPolicyManager

workflow_dir = dirname(dirname(realpath(__file__)))  # template directory
if workflow_dir not in sys.path:
    sys.path.insert(0, workflow_dir)

from general import log_dir, rl_policy_func_index, update_trigger, warmup

def get_policy_manager():
    train_mode = getenv("TRAINMODE", default="single-process")
    policy_dict = {name: func(rollout_only=False) for name, func in rl_policy_func_index.items()}
    if train_mode == "single-process":
        return LocalPolicyManager(
            policy_dict,
            update_trigger=update_trigger,
            warmup=warmup,
            log_dir=log_dir
        )

    num_trainers = int(getenv("NUMTRAINERS", default=5))
    if train_mode == "multi-process":
        return MultiProcessPolicyManager(
            policy_dict,
            num_trainers,
            rl_policy_func_index,
            update_trigger=update_trigger,
            warmup=warmup,
            log_dir=log_dir
        )
    if train_mode == "multi-node":
        return MultiNodePolicyManager(
            policy_dict,
            getenv("TRAINGROUP", default="TRAIN"),
            num_trainers,
            update_trigger=update_trigger,
            warmup=warmup,
            proxy_kwargs={
                "redis_address": (getenv("REDISHOST", default="maro-redis"), int(getenv("REDISPORT", default=6379))),
                "max_peer_discovery_retries": 50    
            },
            log_dir=log_dir
        )

    raise ValueError(
        f"Unsupported policy training mode: {train_mode}. Supported modes: single-process, multi-process, multi-node"
    )
