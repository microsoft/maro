# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os import getenv
from os.path import dirname, realpath

from maro.rl.data_parallelism import grad_worker

workflow_dir = dirname(dirname(realpath(__file__)))  # template directory
if workflow_dir not in sys.path:
    sys.path.insert(0, workflow_dir)

from general import log_dir, policy_func_dict


if __name__ == "__main__":
    # TODO: WORKERID in docker compose script. 
    worker_id = getenv("WORKERID")
    num_hosts = getenv("NUMHOSTS")
    distributed = getenv("DISTRIBUTED") == "True"
    if worker_id is None:
        raise ValueError("missing environment variable: WORKERID")
    if num_hosts is None:
        if distributed:
            raise ValueError("missing environment variable: NUMHOSTS")
        else:
            num_hosts = 0

    group = getenv("POLICYGROUP", default="learn")
    grad_worker(
        policy_func_dict,
        int(worker_id),
        int(num_hosts),
        group,
        proxy_kwargs={
            "redis_address": (getenv("REDISHOST", default="maro-redis"), int(getenv("REDISPORT", default=6379))),
            "max_peer_discovery_retries": 50
        },
        log_dir=log_dir
    )
