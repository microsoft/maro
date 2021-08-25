# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os import getenv
from os.path import dirname, realpath

from maro.rl.learning import DistributedPolicyManager, SimplePolicyManager
from maro.rl.policy import TrainerAllocator
from maro.utils import Logger

workflow_dir = dirname(dirname(realpath(__file__)))  # template directory
if workflow_dir not in sys.path:
    sys.path.insert(0, workflow_dir)

from general import agent2policy, log_dir, rl_policy_func_index

def get_policy_manager():
    logger = Logger("policy manager creator")
    manager_type = getenv("POLICYMANAGERTYPE", default="simple")
    parallel = int(getenv("PARALLEL", default=0))
    if manager_type == "simple":
        return SimplePolicyManager(rl_policy_func_index, parallel=parallel, log_dir=log_dir)

    group = getenv("LEARNGROUP", default="learn")
    num_hosts = int(getenv("NUMHOSTS", default=5))
    if manager_type == "distributed":
        allocation_mode = getenv("ALLOCATIONMODE", default="by-policy")
        allocator = TrainerAllocator(allocation_mode, num_hosts, list(rl_policy_func_index.keys()), agent2policy)
        policy_manager = DistributedPolicyManager(
            rl_policy_func_index, group, num_hosts,
            proxy_kwargs={
                "redis_address": (getenv("REDISHOST", default="maro-redis"), int(getenv("REDISPORT", default=6379))),
                "max_peer_discovery_retries": 50
            },
            trainer_allocator=allocator,
            log_dir=log_dir
        )
        return policy_manager

    raise ValueError(f"Unsupported policy manager type: {manager_type}. Supported modes: simple, distributed")


if __name__ == "__main__":
    policy_manager = get_policy_manager()
    policy_manager.server(
        getenv("GROUP", default="ASYNC"),
        int(getenv("NUMROLLOUTS", default=5)),
        max_lag=int(getenv("MAXLAG", default=0)),
        proxy_kwargs={
            "redis_address": (getenv("REDISHOST", default="maro-redis"), int(getenv("REDISPORT", default=6379))),
            "max_peer_discovery_retries": 50    
        },
        log_dir=log_dir
    )
