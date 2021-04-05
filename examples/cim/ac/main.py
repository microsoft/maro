# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import yaml
from os import getenv
from os.path import dirname, join, realpath

import numpy as np

from maro.rl import (
    Actor, ActorCritic, ActorCriticConfig, FullyConnectedBlock, MultiAgentWrapper, Learner, Scheduler,
    SimpleMultiHeadModel, OptimOption
)
from maro.simulator import Env
from maro.utils import set_seeds

from examples.cim.env_wrapper import CIMEnvWrapper


DEFAULT_CONFIG_PATH = join(dirname(realpath(__file__)), "config.yml")
with open(getenv("CONFIG_PATH", default=DEFAULT_CONFIG_PATH), "r") as config_file:
    config = yaml.safe_load(config_file)

# model input and output dimensions
IN_DIM = (
    (config["shaping"]["look_back"] + 1) *
    (config["shaping"]["max_ports_downstream"] + 1) *
    len(config["shaping"]["port_attributes"]) +
    len(config["shaping"]["vessel_attributes"])
)
OUT_DIM = config["shaping"]["num_actions"]


def get_ac_agent():
    cfg = config["agent"]
    actor_net  = FullyConnectedBlock(input_dim=IN_DIM, output_dim=OUT_DIM, **cfg["model"]["actor"])
    critic_net = FullyConnectedBlock(input_dim=IN_DIM, output_dim=1, **cfg["model"]["critic"])
    ac_model = SimpleMultiHeadModel(
        {"actor": actor_net, "critic": critic_net},
        optim_option={
            "actor":  OptimOption(**cfg["optimization"]["actor"]),
            "critic": OptimOption(**cfg["optimization"]["critic"])
        }
    )
    return ActorCritic(ac_model, ActorCriticConfig(**cfg["algorithm"]), **cfg["experience_memory"])


# Single-threaded launcher
if __name__ == "__main__":
    set_seeds(1024)  # for reproducibility
    env = Env(**config["training"]["env"])
    agent = MultiAgentWrapper({name: get_ac_agent() for name in env.agent_idx_list})
    scheduler = Scheduler(config["training"]["max_episode"])
    learner = Learner(
        CIMEnvWrapper(env, **config["shaping"]), agent, scheduler,
        agent_update_interval=config["training"]["agent_update_interval"]
    ) 
    learner.run()
