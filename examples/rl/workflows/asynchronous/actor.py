# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os import getenv
from os.path import dirname, realpath

from maro.rl.learning import EnvironmentSampler

workflow_dir = dirname(dirname(realpath(__file__)))  # DQN directory  
if workflow_dir not in sys.path:
    sys.path.insert(0, workflow_dir)

from agent_wrapper import get_agent_wrapper
from general import get_env_wrapper, log_dir


if __name__ == "__main__":
    index = getenv("ACTORID")
    if index is None:
        raise ValueError("Missing environment variable: ACTORID")
    index = int(index)

    num_episodes = getenv("NUMEPISODES")
    if num_episodes is None:
        raise ValueError("Missing envrionment variable: NUMEPISODES")
    num_episodes = int(num_episodes)
    num_steps = int(getenv("NUMSTEPS", default=-1))

    env_sampler = EnvironmentSampler(get_env_wrapper, get_agent_wrapper)
    env_sampler.actor(
        getenv("GROUP", default="ASYNC"), index, num_episodes,
        num_steps=num_steps,
        proxy_kwargs={
            "redis_address": (getenv("REDISHOST", default="maro-redis"), int(getenv("REDISPORT", default=6379))),
            "max_peer_discovery_retries": 50
        },
        log_dir=log_dir,
    )
