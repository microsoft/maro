# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import importlib
import os
import sys
from os import getenv

from maro.rl.data_parallelism import task_queue
from maro.rl.workflows.helpers import from_env, get_default_log_dir

sys.path.insert(0, from_env("SCENARIODIR"))
module = importlib.import_module(from_env("SCENARIO"))
policy_func_dict = getattr(module, "policy_func_dict")

log_dir = from_env("LOGDIR", required=False, default=get_default_log_dir(from_env("JOB")))
os.makedirs(log_dir, exist_ok=True)

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
