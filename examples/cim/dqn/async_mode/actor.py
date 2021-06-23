# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

from maro.rl import actor
from maro.simulator import Env

async_mode_path = os.path.dirname(os.path.realpath(__file__))  # DQN async mode directory
dqn_path = os.path.dirname(async_mode_path)  # DQN directory
cim_path = os.path.dirname(dqn_path)  # CIM example directory
sys.path.insert(0, cim_path)
sys.path.insert(0, dqn_path)
sys.path.insert(0, async_mode_path)
from agent_wrapper import get_agent_wrapper
from env_wrapper import CIMEnvWrapper
from general import config, log_dir


if __name__ == "__main__":
    actor(
        os.environ["ACTORID"],
        lambda: CIMEnvWrapper(Env(**config["env"]["basic"]), **config["env"]["wrapper"]),
        get_agent_wrapper,
        config["num_episodes"],
        config["async"]["group"],
        num_steps=config["num_steps"],
        log_dir=log_dir,
    )
