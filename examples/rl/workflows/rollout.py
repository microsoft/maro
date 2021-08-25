# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os import getenv
from os.path import dirname, realpath

from maro.rl.learning import EnvironmentSampler
from maro.rl.wrappers import AgentWrapper

workflow_dir = dirname(dirname(realpath(__file__)))  # template directory
if workflow_dir not in sys.path:
    sys.path.insert(0, workflow_dir)

from general import (
    agent2policy, get_env_wrapper, get_eval_env_wrapper, log_dir, non_rl_policy_func_index, rl_policy_func_index
)


def get_agent_wrapper():
    return AgentWrapper(
        {**non_rl_policy_func_index, **rl_policy_func_index},
        agent2policy
    )


if __name__ == "__main__":
    index = getenv("WORKERID")
    if index is None:
        raise ValueError("Missing environment variable: WORKERID")
    index = int(index)

    env_sampler = EnvironmentSampler(get_env_wrapper, get_agent_wrapper, get_eval_env_wrapper=get_eval_env_wrapper)
    if getenv("MODE") == "sync":
        env_sampler.worker(
            getenv("ROLLOUTGROUP", default="rollout"), index,
            proxy_kwargs={
                "redis_address": (getenv("REDISHOST", default="maro-redis"), int(getenv("REDISPORT", default=6379))),
                "max_peer_discovery_retries": 50    
            },
            log_dir=log_dir
        )
    else:
        num_episodes = getenv("NUMEPISODES")
        if num_episodes is None:
            raise ValueError("Missing envrionment variable: NUMEPISODES")
        num_episodes = int(num_episodes)
        num_steps = int(getenv("NUMSTEPS", default=-1))
        env_sampler.actor(
            getenv("GROUP", default="ASYNC"), index, num_episodes,
            num_steps=num_steps,
            proxy_kwargs={
                "redis_address": (getenv("REDISHOST", default="maro-redis"), int(getenv("REDISPORT", default=6379))),
                "max_peer_discovery_retries": 50
            },
            log_dir=log_dir,
        )
