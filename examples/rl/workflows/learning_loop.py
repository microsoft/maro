# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os import getenv
from os.path import dirname, realpath

from maro.rl.learning import DistributedRolloutManager, MultiProcessRolloutManager, learn

workflow_dir = dirname(dirname((realpath(__file__))))
if workflow_dir not in sys.path:
    sys.path.insert(0, workflow_dir)

from general import post_collect, post_evaluate, get_env_sampler, log_dir


def get_rollout_manager():
    rollout_type = getenv("ROLLOUTTYPE", default="simple")
    num_steps = int(getenv("NUMSTEPS", default=-1))
    if rollout_type == "multi-process":
        return MultiProcessRolloutManager(
            get_env_sampler,
            num_steps=num_steps,
            num_rollouts=int(getenv("NUMROLLOUTS", default="1")),
            num_eval_rollouts=int(getenv("NUMEVALROLLOUTS", default="1")),
            log_dir=log_dir
        )

    num_workers = int(getenv("NUMROLLOUTS", default=5))
    num_eval_workers = int(getenv("NUMEVALROLLOUTS", default=1))
    max_lag = int(getenv("MAXLAG", default=0))
    min_finished_workers = getenv("MINFINISH")
    if min_finished_workers is not None:
        min_finished_workers = int(min_finished_workers)

    max_extra_recv_tries = getenv("MAXEXRECV")
    if max_extra_recv_tries is not None:
        max_extra_recv_tries = int(max_extra_recv_tries)

    extra_recv_timeout = getenv("MAXRECVTIMEO")
    if extra_recv_timeout is not None:
        extra_recv_timeout = int(extra_recv_timeout)

    if rollout_type == "distributed":
        return DistributedRolloutManager(
            getenv("ROLLOUTGROUP", default="rollout"),
            num_workers,
            num_eval_workers=num_eval_workers,
            num_steps=num_steps,
            max_lag=max_lag,
            min_finished_workers=min_finished_workers,
            max_extra_recv_tries=max_extra_recv_tries,
            extra_recv_timeout=extra_recv_timeout,
            proxy_kwargs={
                "redis_address": (getenv("REDISHOST", default="maro-redis"), int(getenv("REDISPORT", default=6379))),
                "max_peer_discovery_retries": 50
            },
        )

    raise ValueError(f"Unsupported roll-out type: {rollout_type}. Supported: simple, distributed")


if __name__ == "__main__":
    num_episodes = getenv("NUMEPISODES")
    if num_episodes is None:
        raise ValueError("Missing environment variable: NUMEPISODES")

    if getenv("MODE") != "single":
        from policy_manager import get_policy_manager
    learn(
        get_rollout_manager if getenv("MODE") != "single" else get_env_sampler,
        int(num_episodes),
        get_policy_manager=get_policy_manager if getenv("MODE") != "single" else None,
        eval_schedule=int(getenv("EVALSCH")),
        post_collect=post_collect,
        post_evaluate=post_evaluate,
        log_dir=log_dir
    )
