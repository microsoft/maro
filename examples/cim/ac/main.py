# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import yaml
from os import getenv
from os.path import dirname, join, realpath

import numpy as np
import torch

from maro.rl import (
    ActorCritic, ActorCriticConfig, AgentWrapper, DiscreteACNet, ExperienceManager, FullyConnectedBlock, LocalLearner,
    OptimOption
)
from maro.simulator import Env
from maro.utils import set_seeds

from examples.cim.env_wrapper import CIMEnvWrapper


DEFAULT_CONFIG_PATH = join(dirname(realpath(__file__)), "config.yml")
with open(getenv("CONFIG_PATH", default=DEFAULT_CONFIG_PATH), "r") as config_file:
    config = yaml.safe_load(config_file)

# model input and output dimensions
IN_DIM = (
    (config["env"]["wrapper"]["look_back"] + 1) *
    (config["env"]["wrapper"]["max_ports_downstream"] + 1) *
    len(config["env"]["wrapper"]["port_attributes"]) +
    len(config["env"]["wrapper"]["vessel_attributes"])
)
OUT_DIM = config["env"]["wrapper"]["num_actions"]


def get_ac_policy(name):
    class MyACNET(DiscreteACNet):
        def forward(self, states, actor: bool = True, critic: bool = True):
            states = torch.from_numpy(np.asarray(states))
            if len(states.shape) == 1:
                states = states.unsqueeze(dim=0)

            states = states.to(self.device)
            return (
                self.component["actor"](states) if actor else None,
                self.component["critic"](states) if critic else None
            )

    cfg = config["policy"]
    ac_net = MyACNET(
        component={
            "actor": FullyConnectedBlock(input_dim=IN_DIM, output_dim=OUT_DIM, **cfg["model"]["network"]["actor"]),
            "critic": FullyConnectedBlock(input_dim=IN_DIM, output_dim=1, **cfg["model"]["network"]["critic"])
        },
        optim_option={
            "actor":  OptimOption(**cfg["model"]["optimization"]["actor"]),
            "critic": OptimOption(**cfg["model"]["optimization"]["critic"])
        }
    )
    experience_manager = ExperienceManager(**cfg["experience_manager"])
    return ActorCritic(name, ac_net, experience_manager, ActorCriticConfig(**cfg["algorithm_config"]))


# Single-threaded launcher
if __name__ == "__main__":
    set_seeds(1024)  # for reproducibility
    env = Env(**config["env"]["basic"])
    env_wrapper = CIMEnvWrapper(env, **config["env"]["wrapper"])
    policies = [get_ac_policy(id_) for id_ in env.agent_idx_list]
    agent2policy = {agent_id: agent_id for agent_id in env.agent_idx_list}
    agent_wrapper = AgentWrapper(policies, agent2policy)
    learner = LocalLearner(env_wrapper, agent_wrapper, 40)  # 40 episodes
    learner.run()
