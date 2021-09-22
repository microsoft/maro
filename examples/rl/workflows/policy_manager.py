# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os import getenv, makedirs
from os.path import dirname, join, realpath

from maro.rl.learning import DistributedPolicyManager, MultiProcessPolicyManager, SimplePolicyManager
from maro.rl.policy import WorkerAllocator

workflow_dir = dirname((realpath(__file__)))  # template directory
if workflow_dir not in sys.path:
    sys.path.insert(0, workflow_dir)

from general import agent2policy, log_dir, policy_func_dict

checkpoint_dir = join(getenv("CHECKPOINTDIR"), getenv("JOB"))
makedirs(checkpoint_dir, exist_ok=True)

def get_policy_manager():
    manager_type = getenv("POLICYMANAGERTYPE", default="simple")
    data_parallel = getenv("DATAPARALLEL") == "True"
    num_grad_workers = int(getenv("NUMGRADWORKERS", default=1))
    allocation_mode = getenv("ALLOCATIONMODE", default="by-policy")
    if data_parallel:
        allocator = WorkerAllocator(allocation_mode, num_grad_workers, list(policy_func_dict.keys()), agent2policy)
    else:
        allocator = None
    proxy_kwargs = {
        "redis_address": (getenv("REDISHOST", default="maro-redis"), int(getenv("REDISPORT", default=6379))),
        "max_peer_discovery_retries": 50
    }
    if manager_type == "simple":
        return SimplePolicyManager(
            policy_func_dict, 
            load_path_dict={id_: join(checkpoint_dir, id_) for id_ in policy_func_dict},
            checkpoint_every=7,
            save_dir=checkpoint_dir,
            worker_allocator=allocator,
            group=getenv("POLICYGROUP"),
            proxy_kwargs=proxy_kwargs,
            log_dir=log_dir
        )
    elif manager_type == "multi-process":
        return MultiProcessPolicyManager(
            policy_func_dict, 
            load_path_dict={id_: join(checkpoint_dir, id_) for id_ in policy_func_dict},
            auto_checkpoint=True,
            save_dir=checkpoint_dir,
            worker_allocator=allocator,
            group=getenv("POLICYGROUP"),
            proxy_kwargs=proxy_kwargs,
            log_dir=log_dir
        )
    elif manager_type == "distributed":
        num_hosts = int(getenv("NUMHOSTS", default=5))
        return DistributedPolicyManager(
            list(policy_func_dict.keys()), num_hosts,
            group=getenv("POLICYGROUP"),
            worker_allocator=allocator,
            proxy_kwargs=proxy_kwargs,
            log_dir=log_dir
        )

    raise ValueError(
        f"Unsupported policy manager type: {manager_type}. Supported modes: simple, multi-process, distributed"
    )


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
