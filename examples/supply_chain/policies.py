# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os.path import dirname, realpath

import numpy as np

import torch

from maro.rl import (
    DQN, DQNConfig, EpisodeBasedSchedule, FullyConnectedBlock, NullPolicy, OptimOption, QNetForDiscreteActionSpace,
    StepBasedSchedule, UniformSampler
)

sc_code_dir = dirname(realpath(__file__))
sys.path.insert(0, sc_code_dir)
from config import config

agent_ids = config["agent_ids"]
policy_ids = ["consumer", "producer", "facility", "product"]


class SimpleQNet(QNetForDiscreteActionSpace):
    def forward(self, states):
        states = torch.from_numpy(np.asarray(states)).to(self.device)
        if len(states.shape) == 1:
            states = states.unsqueeze(dim=0)
        return self.component.forward(states)


def get_dqn_policy(cfg):
    q_net = SimpleQNet(
        FullyConnectedBlock(**cfg["model"]["network"]),
        optim_option=OptimOption(**cfg["model"]["optimization"]),
        device=cfg["model"]["device"]
    )
    experience_manager = UniformSampler(**cfg["experience_manager"])
    return DQN(q_net, experience_manager, DQNConfig(**cfg["algorithm_config"]))


null_policy = NullPolicy()
policy_dict = {
    policy_id: get_dqn_policy(config["policy"][policy_id]) if policy_id in config["policy"] else null_policy
    for policy_id in policy_ids
}

agent2policy = {agent_id: agent_id.split(".")[0] for agent_id in agent_ids}

# update schedules
schedule_type = {"step": StepBasedSchedule, "episode": EpisodeBasedSchedule}

def get_policy_update_schedule(cfg):
    return schedule_type[cfg["type"]](**cfg["args"])

# policy update schedule can be a dict or single EpisodeBasedSchedule or StepBasedSchedule.
# The latter indicates that all policies shared the same update schedule 
policy_update_schedule = {
    policy_id: get_policy_update_schedule(config["policy"][policy_id]["update_schedule"])
    for policy_id in policy_ids if policy_id in config["policy"]
}