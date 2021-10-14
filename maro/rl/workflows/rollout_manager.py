# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import importlib
import os
import sys

from maro.rl.learning import DistributedRolloutManager, MultiProcessRolloutManager
from maro.rl.workflows.helpers import from_env, get_default_log_dir

sys.path.insert(0, from_env("SCENARIODIR"))
module = importlib.import_module(from_env("SCENARIO"))
get_env_sampler = getattr(module, "get_env_sampler")
post_collect = getattr(module, "post_collect", None)
post_evaluate = getattr(module, "post_evaluate", None)

log_dir = from_env("LOGDIR", required=False, default=get_default_log_dir(from_env("JOB")))
os.makedirs(log_dir, exist_ok=True)


def get_rollout_manager():
    rollout_type = from_env("ROLLOUTTYPE")
    num_steps = from_env("NUMSTEPS", required=False, default=-1)
    if rollout_type == "multi-process":
        return MultiProcessRolloutManager(
            get_env_sampler,
            num_steps=num_steps,
            num_rollouts=from_env("NUMROLLOUTS"),
            num_eval_rollouts=from_env("NUMEVALROLLOUTS", required=False, default=1),
            log_dir=log_dir
        )

    if rollout_type == "distributed":
        num_workers = from_env("NUMROLLOUTS")
        num_eval_workers = from_env("NUMEVALROLLOUTS", required=False, default=1)
        min_finished_workers = from_env("MINFINISH", required=False, default=None)
        max_extra_recv_tries = from_env("MAXEXRECV", required=False, default=0)
        extra_recv_timeout = from_env("MAXRECVTIMEO", required=False, default=100)

        return DistributedRolloutManager(
            from_env("ROLLOUTGROUP"),
            num_workers,
            num_eval_workers=num_eval_workers,
            num_steps=num_steps,
            min_finished_workers=min_finished_workers,
            max_extra_recv_tries=max_extra_recv_tries,
            extra_recv_timeout=extra_recv_timeout,
            proxy_kwargs={
                "redis_address": (from_env("REDISHOST"), from_env("REDISPORT")),
                "max_peer_discovery_retries": 50
            },
        )

    raise ValueError(f"Unsupported roll-out type: {rollout_type}. Supported: multi-process, distributed")
