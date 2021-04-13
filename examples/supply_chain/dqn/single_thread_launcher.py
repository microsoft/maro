# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import yaml
from os import getenv
from os.path import dirname, join, realpath

import numpy as np

from maro.rl import Learner, LinearParameterScheduler
from maro.simulator import Env
from maro.utils import set_seeds

sc_code_dir = dirname(dirname(__file__))
sys.path.insert(0, sc_code_dir)
sys.path.insert(0, join(sc_code_dir, "dqn"))
from env_wrapper import SCEnvWrapper
from agent import get_dqn_agent


# Single-threaded launcher
if __name__ == "__main__":
    defualt = join(dirname(realpath(__file__)), "config.yml")
    with open(getenv("CONFIG_PATH", default=default_config_path), "r") as config_file:
        config = yaml.safe_load(config_file)

    # Get the env config path
    topology = config["training"]["env"]["topology"]
    config["training"]["env"]["topology"] = join(dirname(dirname(realpath(__file__))), "envs", topology)

    # create an env wrapper and obtain the input dimension for the pro
    env = SCEnvWrapper(Env(**config["training"]["env"]))
    
    # create agents
    agent_info_list = env.agent_idx_list
    producers = {f"producer.{info.id}": get_dqn_agent(config["agent"]["producer"]) for info in agent_info_list}
    consumers = {f"consumer.{info.id}": get_dqn_agent(config["agent"]["consumer"]) for info in agent_info_list}
    agent = MultiAgentWrapper({**producers, **consumers})

    # exploration schedule
    scheduler = LinearParameterScheduler(config["training"]["max_episode"], **config["training"]["exploration"])    

    # create a learner to start training
    learner = Learner(env, agent, scheduler, agent_update_interval=config["training"]["agent_update_interval"])
    learner.run()
