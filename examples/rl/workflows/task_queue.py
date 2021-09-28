# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os import getenv
from os.path import dirname, realpath

from maro.rl.learning.task_queue import task_queue

workflow_dir = dirname(dirname(realpath(__file__)))  # template directory
if workflow_dir not in sys.path:
    sys.path.insert(0, workflow_dir)

from general import log_dir, policy_func_dict


if __name__ == "__main__":
    num_hosts = getenv("NUMHOSTS")
    data_parallel = getenv("DATAPARALLEL") == "True"
    num_grad_workers = getenv("NUMGRADWORKERS")

    if num_grad_workers is None:
        num_grad_workers = 0
    if num_hosts is None:
        # in multi-process or simple mode
        num_hosts = 0

    # type convert
    num_hosts = int(num_hosts)
    num_grad_workers = int(num_grad_workers)

    worker_id_list = [f"GRAD_WORKER.{i}" for i in range(num_grad_workers)]

    group = getenv("POLICYGROUP", default="learn")
    task_queue(
        worker_id_list,
        num_hosts,
        len(policy_func_dict),
        group=group,
        proxy_kwargs={
            "redis_address": (getenv("REDISHOST", default="maro-redis"), int(getenv("REDISPORT", default=6379))),
            "max_peer_discovery_retries": 50
        },
        log_dir=log_dir
    )
