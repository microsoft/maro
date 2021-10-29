# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import importlib
import os
import sys

from maro.rl.data_parallelism import task_queue
from maro.rl.workflows.helpers import from_env, get_default_log_dir

sys.path.insert(0, from_env("SCENARIODIR"))
module = importlib.import_module(from_env("SCENARIO"))
policy_func_dict = getattr(module, "policy_func_dict")

log_dir = from_env("LOGDIR", required=False, default=get_default_log_dir(from_env("JOB")))
os.makedirs(log_dir, exist_ok=True)

if __name__ == "__main__":
    num_hosts = from_env("NUMHOSTS", required=False, default=0)
    data_parallelism = from_env("DATAPARALLELISM", required=False, default=1)
    single_task_limit = float(os.getenv("SINGLETASKLIMIT", default=0.5))

    worker_id_list = [f"GRAD_WORKER.{i}" for i in range(data_parallelism)]

    group = from_env("POLICYGROUP", required=False, default="learn")
    task_queue(
        worker_id_list,
        num_hosts,
        len(policy_func_dict),
        single_task_limit=single_task_limit,
        group=group,
        proxy_kwargs={
            "redis_address": (
                from_env("REDISHOST", required=False, default="maro-redis"),
                from_env("REDISPORT", required=False, default=6379)),
            "max_peer_discovery_retries": 50
        },
        log_dir=log_dir
    )
