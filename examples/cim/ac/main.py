# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import yaml
from os import getenv
from os.path import dirname, join, realpath

import numpy as np

from maro.rl import (
    Actor, ActorCritic, ActorCriticConfig, FullyConnectedBlock, MultiAgentWrapper, SimpleMultiHeadModel,
    OnPolicyLearner, OptimOption
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
    actor_net  = FullyConnectedBlock(input_dim=IN_DIM, output_dim=OUT_DIM, **config["agent"]["model"]["actor"])
    critic_net = FullyConnectedBlock(input_dim=IN_DIM, output_dim=1, **config["agent"]["model"]["critic"])
    ac_model = SimpleMultiHeadModel(
        {"actor": actor_net, "critic": critic_net},
        optim_option={
            "actor":  OptimOption(**config["agent"]["optimization"]["actor"]),
            "critic": OptimOption(**config["agent"]["optimization"]["critic"])
        }
    )
    return ActorCritic(ac_model, ActorCriticConfig(**config["agent"]["hyper_params"]))


# Single-threaded launcher
if __name__ == "__main__":
    set_seeds(1024)  # for reproducibility
    env = Env(**config["training"]["env"])
    agent = MultiAgentWrapper({name: get_ac_agent() for name in env.agent_idx_list})
    learner = OnPolicyLearner(CIMEnvWrapper(env, **config["shaping"]), agent, config["training"]["max_episode"])
    learner.run()
