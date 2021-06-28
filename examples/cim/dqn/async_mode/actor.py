# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

from maro.rl import actor
from maro.simulator import Env

dqn_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # DQN directory  
cim_path = os.path.dirname(dqn_path)  # CIM example directory
sys.path.insert(0, cim_path)
sys.path.insert(0, dqn_path)
from agent_wrapper import get_agent_wrapper
from env_wrapper import CIMEnvWrapper
from general import config, log_dir


if __name__ == "__main__":
    actor(
        config["async"]["group"],
        CIMEnvWrapper(Env(**config["env"]["basic"]), **config["env"]["wrapper"]),
        get_agent_wrapper(),
        config["num_episodes"],
        num_steps=config["num_steps"],
        proxy_kwargs={
            "redis_address": (config["redis"]["host"], config["redis"]["port"]),
            "component_name": os.environ["ACTORID"],
        },
        log_dir=log_dir,
    )
