# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import yaml
from os import getenv
from os.path import dirname, join, realpath

import numpy as np

from maro.rl import (
    AGENT_CLS, AGENT_CONFIG, AgentGroup, AgentManager, FullyConnectedBlock, GenericAgentConfig, Learner,
    LinearParameterScheduler, OptimOption, SimpleMultiHeadModel, get_cls
)
from maro.simulator import Env
from maro.utils import set_seeds

sc_code_dir = dirname(realpath(__file__))
sys.path.insert(0, sc_code_dir)
from env_wrapper import SCEnvWrapper

# Single-threaded launcher
if __name__ == "__main__":
    default_config_path = join(dirname(realpath(__file__)), "config.yml")
    with open(getenv("CONFIG_PATH", default=default_config_path), "r") as config_file:
        config = yaml.safe_load(config_file)

    # Get the env config path
    topology = config["training"]["env"]["topology"]
    config["training"]["env"]["topology"] = join(dirname(realpath(__file__)), "topologies", topology)

    # create an env wrapper and obtain the input dimension for the agents
    env = SCEnvWrapper(Env(**config["training"]["env"]))
    config["agent"]["producer"]["model"]["network"]["input_dim"] = env.dim
    config["agent"]["consumer"]["model"]["network"]["input_dim"] = env.dim

    # create agents
    agent_names = [info.id for info in env.agent_idx_list]

    def get_model(model_config):
        return SimpleMultiHeadModel(
            FullyConnectedBlock(**model_config["network"]),
            optim_option=OptimOption(**model_config["optimization"])
        )

    def get_sc_agents(agent_names, agent_config):
        agent_cls = get_cls(agent_config["algorithm"], AGENT_CLS)
        algorithm_config_cls = get_cls(agent_config["algorithm"], AGENT_CONFIG)
        algorithm_config = algorithm_config_cls(**agent_config["algorithm_config"])
        generic_config = GenericAgentConfig(**agent_config["generic_config"])
        if agent_config["share_model"]:
            shared_model = get_model(agent_config["model"])
            return AgentGroup(agent_names, agent_cls, shared_model, algorithm_config, generic_config)
        else:
            return AgentManager({
                name: agent_cls(get_model(agent_config["model"]), algorithm_config, generic_config)
                for name in agent_names
            })

    agent = AgentManager({
        "producer": get_sc_agents(agent_names, config["agent"]["producer"]),
        "consumer": get_sc_agents(agent_names, config["agent"]["consumer"])
    })

    # exploration schedule
    scheduler = LinearParameterScheduler(config["training"]["num_episodes"], **config["training"]["exploration"])    

    # create a learner to start training
    learner = Learner(env, agent, scheduler, agent_update_interval=config["training"]["agent_update_interval"])
    learner.run()
