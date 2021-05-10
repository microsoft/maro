# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os.path import dirname, realpath

import numpy as np

import torch

from maro.rl import (
    DQN, DQNConfig, EpisodeBasedSchedule, FullyConnectedBlock, NullPolicy,
    OptimOption, QNetForDiscreteActionSpace, StepBasedSchedule, UniformSampler
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
    q_net = SimpleQNet(
        FullyConnectedBlock(**config["model"]["network"]),
        optim_option=OptimOption(**config["model"]["optimization"]),
        device=config["model"]["device"]
    )
    experience_manager = UniformSampler(**config["experience_manager"])
    return DQN(q_net, experience_manager, DQNConfig(**config["algorithm_config"]))

# all consumers share the same underlying policy
policy_dict = {
    "consumer": get_dqn_policy(config["consumer"]),
    "producer": get_dqn_policy(config["consumer"]),
    "facility": NullPolicy(),
    "product": NullPolicy()
}

agent2policy = {agent_id: agent_id.split(".")[0] for agent_id in agent_ids}
