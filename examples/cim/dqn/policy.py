# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from maro.rl import DQN, DQNConfig, DiscreteQNet, ExperienceManager, FullyConnectedBlock, OptimOption

dqn_path = os.path.dirname(os.path.realpath(__file__))  # DQN directory
cim_path = os.path.dirname(dqn_path)  # CIM example directory
sys.path.insert(0, cim_path)
sys.path.insert(0, dqn_path)
from general import config


class QNet(DiscreteQNet):
    def __init__(self, component: nn.Module, optim_option: OptimOption=None, device=None):
        super().__init__(component, optim_option=optim_option, device=device)

    def forward(self, states):
        states = torch.from_numpy(np.asarray(states)).to(self.device)
        if len(states.shape) == 1:
            states = states.unsqueeze(dim=0)
        return self.component(states)


def get_independent_policy_for_training(name):
    cfg = config["policies"]
    qnet = QNet(
        FullyConnectedBlock(**cfg["model"]["network"]),
        optim_option=OptimOption(**cfg["model"]["optimization"])
    )
    return DQN(
        name=name,
        q_net=qnet,
        experience_manager=ExperienceManager(**cfg["experience_manager"]["training"]),
        config=DQNConfig(**cfg["algorithm_config"]),
        update_trigger=cfg["update_trigger"],
        warmup=cfg["warmup"]
    )


def get_independent_policy_for_rollout(name):
    cfg = config["policies"]
    qnet = QNet(FullyConnectedBlock(**cfg["model"]["network"]))
    return DQN(
        name=name,
        q_net=qnet,
        experience_manager=ExperienceManager(**cfg["experience_manager"]["rollout"]),
        config=DQNConfig(**cfg["algorithm_config"])
    )
