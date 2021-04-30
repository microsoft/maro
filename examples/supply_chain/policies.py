# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os.path import dirname, realpath

import numpy as np

import torch

from maro.rl import (
    DQN, DQNConfig, ExperienceMemory, FullyConnectedBlock, NullPolicy, OptimOption, QNetForDiscreteActionSpace,
    TrainingLoopConfig, get_sampler_cls
)

sc_code_dir = dirname(realpath(__file__))
sys.path.insert(0, sc_code_dir)
from config import config

agent_ids = config["agent_ids"]
config = config["policy"]


class SimpleQNet(QNetForDiscreteActionSpace):
    def forward(self, states):
        states = torch.from_numpy(np.asarray(states)).to(self.device)
        if len(states.shape) == 1:
            states = states.unsqueeze(dim=0)
        return self.component.forward(states)


def get_dqn_policy(config):
    q_net = SimpleQNet(FullyConnectedBlock(**config["model"]), optim_option=OptimOption(**config["optimization"]))
    experience_memory = ExperienceMemory(**config["experience_memory"])

    config["training_loop"]["sampler_cls"] = get_sampler_cls(config["training_loop"]["sampler_cls"])
    generic_config = TrainingLoopConfig(**config["training_loop"])
    special_config = DQNConfig(**config["algorithm_config"])

    return DQN(q_net, experience_memory, generic_config, special_config)

# all consumers share the same underlying policy
policy_dict = {"consumer": get_dqn_policy(config["consumer"]), "producer": NullPolicy()}

agent_to_policy = {agent_id: agent_id.split(".")[0] for agent_id in agent_ids}
