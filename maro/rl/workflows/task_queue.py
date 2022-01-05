# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from maro.rl.data_parallelism import task_queue
from maro.rl.workflows.helpers import from_env, get_logger, get_scenario_module

if __name__ == "__main__":
    num_hosts = from_env("NUM_HOSTS", required=False, default=0)
    policy_func_dict = getattr(get_scenario_module(from_env("SCENARIO_PATH")), "policy_func_dict")
    data_parallelism = from_env("DATA_PARALLELISM", required=False, default=1)
    worker_id_list = [f"GRAD_WORKER.{i}" for i in range(data_parallelism)]

    task_queue(
        worker_id_list,
        num_hosts,
        len(policy_func_dict),
        single_task_limit=from_env("SINGLETASKLIMIT", required=False, default=0.5),
        group=from_env("POLICY_GROUP", required=False, default="learn"),
        proxy_kwargs={
            "redis_address": (
                from_env("REDIS_HOST", required=False, default="maro-redis"),
                from_env("REDIS_PORT", required=False, default=6379)),
            "max_peer_discovery_retries": 50
        },
        logger=get_logger(from_env("LOG_PATH", required=False, default=os.getcwd()), from_env("JOB"), "TASK_QUEUE")
    )
