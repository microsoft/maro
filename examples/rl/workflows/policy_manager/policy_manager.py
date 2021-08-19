# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os import getenv
from os.path import dirname, realpath

from maro.rl.learning import DistributedPolicyManager, SimplePolicyManager

workflow_dir = dirname(dirname(realpath(__file__)))  # template directory
if workflow_dir not in sys.path:
    sys.path.insert(0, workflow_dir)

from general import log_dir, rl_policy_func_index

def get_policy_manager():
    manager_type = getenv("POLICYMANAGERTYPE", default="simple")
    parallel = int(getenv("PARALLEL", default=0))
    if manager_type == "simple":
        return SimplePolicyManager(rl_policy_func_index, parallel=parallel, log_dir=log_dir)

    num_hosts = int(getenv("NUMHOSTS", default=5))
    if manager_type == "distributed":
        return DistributedPolicyManager(
            list(rl_policy_func_index.keys()),
            getenv("LEARNGROUP", default="learn"),
            num_hosts,
            proxy_kwargs={
                "redis_address": (getenv("REDISHOST", default="maro-redis"), int(getenv("REDISPORT", default=6379))),
                "max_peer_discovery_retries": 50
            },
            log_dir=log_dir
        )
    if train_mode == "multi-node-dist":
        allocator = TrainerAllocator(allocation_mode, num_trainers, list(policy_dict.keys()), agent2policy)
        return MultiNodeDistPolicyManager(
            policy_dict,
            update_option,
            getenv("TRAINGROUP", default="TRAIN"),
            num_trainers,
            proxy_kwargs={
                "redis_address": (getenv("REDISHOST", default="maro-redis"), int(getenv("REDISPORT", default=6379))),
                "max_peer_discovery_retries": 50
            },
            trainer_allocator=allocator,
            log_dir=log_dir
        )

    raise ValueError(f"Unsupported policy manager type: {manager_type}. Supported modes: simple, distributed")
