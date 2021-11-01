# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from maro.rl.learning import DistributedRolloutManager, MultiProcessRolloutManager
from maro.rl.workflows.helpers import from_env, get_log_dir, get_scenario_module


def get_rollout_manager():
    rollout_type = from_env("ROLLOUTTYPE")
    num_steps = from_env("NUMSTEPS", required=False, default=-1)
    log_dir = get_log_dir(from_env("LOGDIR", required=False, default=os.getcwd()), from_env("JOB"))
    if rollout_type == "multi-process":
        return MultiProcessRolloutManager(
            getattr(get_scenario_module(from_env("SCENARIODIR")), "get_env_sampler"),
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
            log_dir=log_dir
        )

    raise ValueError(f"Unsupported roll-out type: {rollout_type}. Supported: multi-process, distributed")
