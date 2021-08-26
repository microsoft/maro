# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os import getenv
from os.path import dirname, realpath

from maro.rl.learning.synchronous import (
    Learner, LocalRolloutManager, MultiNodeRolloutManager, MultiProcessRolloutManager
)

workflow_dir = dirname(dirname((realpath(__file__))))
if workflow_dir not in sys.path:
    sys.path.insert(0, workflow_dir)

from agent_wrapper import get_agent_wrapper
from policy_manager.policy_manager import get_policy_manager
from general import post_collect, post_evaluate, get_env_wrapper, log_dir, replay_agents


def get_rollout_manager():
    rollout_mode = getenv("ROLLOUTMODE", default="single-process")
    num_steps = int(getenv("NUMSTEPS", default=-1))
    if rollout_mode == "single-process":
        return LocalRolloutManager(
            get_env_wrapper(replay_agent_ids=replay_agents[0]),
            get_agent_wrapper(),
            num_steps=num_steps,
            post_collect=post_collect,
            post_evaluate=post_evaluate,
            log_dir=log_dir
        )

    num_workers = int(getenv("NUMWORKERS", default=5))
    if rollout_mode == "multi-process":
        return MultiProcessRolloutManager(
            num_workers,
            get_env_wrapper,
            get_agent_wrapper,
            num_steps=num_steps,
            post_collect=post_collect,
            post_evaluate=post_evaluate,
            log_dir=log_dir,
        )

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

    if rollout_mode == "multi-node":
        return MultiNodeRolloutManager(
            getenv("ROLLOUTGROUP", default="ROLLOUT"),
            num_workers,
            num_steps=num_steps,
            max_lag=max_lag,
            min_finished_workers=min_finished_workers,
            max_extra_recv_tries=max_extra_recv_tries,
            extra_recv_timeout=extra_recv_timeout,
            post_collect=post_collect,
            post_evaluate=post_evaluate,
            proxy_kwargs={
                "redis_address": (getenv("REDISHOST", default="maro-redis"), int(getenv("REDISPORT", default=6379))),
                "max_peer_discovery_retries": 50
            },
        )

    raise ValueError(
        f"Unsupported roll-out mode: {rollout_mode}. Supported modes: single-process, multi-process, multi-node"
    )


if __name__ == "__main__":
    num_episodes = getenv("NUMEPISODES")
    if num_episodes is None:
        raise ValueError("Missing envrionment variable: NUMEPISODES")

    learner = Learner(
        get_policy_manager(),
        get_rollout_manager(),
        int(num_episodes),
        eval_schedule=int(getenv("EVALSCH")),
        log_dir=log_dir
    )
    learner.run()
