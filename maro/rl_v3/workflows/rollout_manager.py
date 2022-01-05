# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from maro.rl.learning import DistributedRolloutManager, MultiProcessRolloutManager
from maro.rl_v3.utils.common import from_env, get_logger, get_module


def get_rollout_manager():
    rollout_type = from_env("ROLLOUT_TYPE")
    num_steps = from_env("NUM_STEPS", required=False, default=-1)
    logger = get_logger(from_env("LOG_PATH", required=False, default=os.getcwd()), from_env("JOB"), "ROLLOUT_MANAGER")
    if rollout_type == "multi-process":
        return MultiProcessRolloutManager(
            getattr(get_module(from_env("SCENARIO_PATH")), "get_env_sampler"),
            num_steps=num_steps,
            num_rollouts=from_env("NUM_ROLLOUTS"),
            num_eval_rollouts=from_env("NUM_EVAL_ROLLOUTS", required=False, default=1),
            logger=logger
        )

    if rollout_type == "distributed":
        num_workers = from_env("NUM_ROLLOUTS")
        num_eval_workers = from_env("NUM_EVAL_ROLLOUTS", required=False, default=1)
        min_finished_workers = from_env("MIN_FINISHED_WORKERS", required=False, default=None)
        max_extra_recv_tries = from_env("MAX_EXTRA_RECV_TRIES", required=False, default=0)
        extra_recv_timeout = from_env("MAX_RECV_TIMEO", required=False, default=100)

        return DistributedRolloutManager(
            from_env("ROLLOUT_GROUP"),
            num_workers,
            num_eval_workers=num_eval_workers,
            num_steps=num_steps,
            min_finished_workers=min_finished_workers,
            max_extra_recv_tries=max_extra_recv_tries,
            extra_recv_timeout=extra_recv_timeout,
            proxy_kwargs={
                "redis_address": (from_env("REDIS_HOST"), from_env("REDIS_PORT")),
                "max_peer_discovery_retries": 50
            },
            logger=logger
        )

    raise ValueError(f"Unsupported roll-out type: {rollout_type}. Supported: multi-process, distributed")
