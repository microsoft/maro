# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import yaml
from os import getenv
from os.path import dirname, join, realpath

import numpy as np

from maro.rl import Learner, LinearParameterScheduler, MultiAgentWrapper
from maro.simulator import Env
from maro.utils import set_seeds

sc_code_dir = dirname(realpath(__file__))
sys.path.insert(0, sc_code_dir)
from env_wrapper import SCEnvWrapper
from agent import get_agent_func_map


# Single-threaded launcher
if __name__ == "__main__":
    default_config_path = join(dirname(realpath(__file__)), "config.yml")
    with open(getenv("CONFIG_PATH", default=default_config_path), "r") as config_file:
        config = yaml.safe_load(config_file)

    get_producer_agent = get_agent_func_map[config["agent"]["producer"]["algorithm"]]
    get_consumer_agent = get_agent_func_map[config["agent"]["consumer"]["algorithm"]]

    # Get the env config path
    topology = config["training"]["env"]["topology"]
    config["training"]["env"]["topology"] = join(dirname(realpath(__file__)), "envs", topology)

    # create an env wrapper and obtain the input dimension for the agents
    env = SCEnvWrapper(Env(**config["training"]["env"]))
    config["agent"]["producer"]["model"]["input_dim"] = config["agent"]["consumer"]["model"]["input_dim"] = env.dim

    # create agents
    agent_info_list = env.agent_idx_list
    producers = {f"producer.{info.id}": get_producer_agent(config["agent"]["producer"]) for info in agent_info_list}
    consumers = {f"consumer.{info.id}": get_consumer_agent(config["agent"]["consumer"]) for info in agent_info_list}
    agent = MultiAgentWrapper({**producers, **consumers})

    # exploration schedule
    scheduler = LinearParameterScheduler(config["training"]["max_episode"], **config["training"]["exploration"])    

    # create a learner to start training
    learner = Learner(env, agent, scheduler, agent_update_interval=config["training"]["agent_update_interval"])
    learner.run()
