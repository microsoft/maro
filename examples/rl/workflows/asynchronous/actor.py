# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os import environ
from os.path import dirname, realpath

from maro.rl.learning.asynchronous import actor

workflow_dir = dirname(dirname(realpath(__file__)))  # DQN directory  
if workflow_dir not in sys.path:
    sys.path.insert(0, workflow_dir)

from agent_wrapper import get_agent_wrapper
from general import config, get_env_wrapper, log_dir, replay_agents


if __name__ == "__main__":
    actor_id = int(environ["ACTORID"])
    actor(
        config["async"]["group"],
        actor_id,
        get_env_wrapper(replay_agent_ids=replay_agents[actor_id]),
        get_agent_wrapper(),
        config["num_episodes"],
        num_steps=config["num_steps"],
        proxy_kwargs={"redis_address": (config["redis"]["host"], config["redis"]["port"])},
        log_dir=log_dir,
    )
