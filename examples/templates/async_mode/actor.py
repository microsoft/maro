# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os import environ
from os.path import dirname, realpath

from maro.rl import actor

template_dir = dirname(dirname(realpath(__file__)))  # DQN directory  
if template_dir not in sys.path:
    sys.path.insert(0, template_dir)
from general import config, get_agent_wrapper, get_env_wrapper, log_dir


if __name__ == "__main__":
    actor(
        config["async"]["group"],
        environ["ACTORID"],
        get_env_wrapper(),
        get_agent_wrapper(),
        config["num_episodes"],
        num_steps=config["num_steps"],
        proxy_kwargs={"redis_address": (config["redis"]["host"], config["redis"]["port"])},
        log_dir=log_dir,
    )
