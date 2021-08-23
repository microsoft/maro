# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os import getenv
from os.path import dirname, realpath

from maro.rl.learning import Learner, DistributedRolloutManager, SimpleRolloutManager

workflow_dir = dirname(dirname((realpath(__file__))))
if workflow_dir not in sys.path:
    sys.path.insert(0, workflow_dir)

from agent_wrapper import get_agent_wrapper
from policy_manager.policy_manager import get_policy_manager
from general import post_collect, post_evaluate, get_env_wrapper, get_eval_env_wrapper, log_dir


def get_rollout_manager():
    rollout_type = getenv("ROLLOUTTYPE", default="simple")
    num_steps = int(getenv("NUMSTEPS", default=-1))
    if rollout_type == "simple":
        return SimpleRolloutManager(
            get_env_wrapper,
            get_agent_wrapper,
            get_eval_env_wrapper=get_eval_env_wrapper,
            num_steps=num_steps,
            parallelism=int(getenv("PARALLELISM", default="1")),
            eval_parallelism=int(getenv("EVALPARALLELISM", default="1")),
            post_collect=post_collect,
            post_evaluate=post_evaluate,
            log_dir=log_dir
        )

    num_workers = int(getenv("NUMROLLOUTS", default=5))
    num_eval_workers = int(getenv("NUMEVALWORKERS", default=1))
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
            post_collect=post_collect,
            post_evaluate=post_evaluate,
            proxy_kwargs={
                "redis_address": (getenv("REDISHOST", default="maro-redis"), int(getenv("REDISPORT", default=6379))),
                "max_peer_discovery_retries": 50
            },
        )

    raise ValueError(f"Unsupported roll-out type: {rollout_type}. Supported: simple, distributed")


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
