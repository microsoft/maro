# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os import getenv
from os.path import dirname, realpath

from maro.rl.learning.asynchronous import actor

workflow_dir = dirname(dirname(realpath(__file__)))  # DQN directory  
if workflow_dir not in sys.path:
    sys.path.insert(0, workflow_dir)

from agent_wrapper import get_agent_wrapper
from general import get_env_wrapper, get_eval_env_wrapper, log_dir, replay_agents


if __name__ == "__main__":
    actor_id = getenv("ACTORID")
    if actor_id is None:
        raise ValueError("Missing environment variable: ACTORID")
    actor_id = int(actor_id)

    num_episodes = getenv("NUMEPISODES")
    if num_episodes is None:
        raise ValueError("Missing envrionment variable: NUMEPISODES")
    num_episodes = int(num_episodes)
    num_steps = int(getenv("NUMSTEPS", default=-1))

    actor(
        getenv("GROUP", default="ASYNC"),
        actor_id,
        get_env_wrapper(replay_agent_ids=replay_agents[actor_id]),
        get_agent_wrapper(),
        num_episodes,
        num_steps=num_steps,
        eval_env_wrapper=get_eval_env_wrapper(),
        eval_schedule=int(getenv("EVALSCH")),
        proxy_kwargs={
            "redis_address": (getenv("REDISHOST", default="maro-redis"), int(getenv("REDISPORT", default=6379))),
            "max_peer_discovery_retries": 50
        },
        log_dir=log_dir,
    )
