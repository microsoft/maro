# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os import getenv
from os.path import dirname, realpath

from maro.rl.learning import policy_host

workflow_dir = dirname(dirname(realpath(__file__)))  # template directory
if workflow_dir not in sys.path:
    sys.path.insert(0, workflow_dir)

from general import log_dir, policy_func_dict


if __name__ == "__main__":
    host_id = getenv("HOSTID")
    data_parallel = getenv("DATAPARALLEL") == "True"
    num_grad_workers = getenv("NUMGRADWORKERS")

    if host_id is None:
        raise ValueError("missing environment variable: HOSTID")
    if num_grad_workers is None:
        num_grad_workers = 0

    group = getenv("POLICYGROUP", default="learn")
    policy_host(
        policy_func_dict,
        int(host_id),
        group,
        proxy_kwargs={
            "redis_address": (getenv("REDISHOST", default="maro-redis"), int(getenv("REDISPORT", default=6379))),
            "max_peer_discovery_retries": 50
        },
        data_parallel=data_parallel,
        num_grad_workers=int(num_grad_workers),
        log_dir=log_dir
    )
