# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

import numpy as np
import torch
import torch.nn as nn

from maro.rl.exploration import EpsilonGreedyExploration, MultiPhaseLinearExplorationScheduler
from maro.rl.model import FullyConnectedBlock, OptimOption
from maro.rl.policy import DQN, DiscreteQNet

cim_path = os.path.dirname(os.path.realpath(__file__))
if cim_path not in sys.path:
    sys.path.insert(0, cim_path)
from env_wrapper import STATE_DIM, env_config

model_config = {
    "network": {
        "input_dim": STATE_DIM,
        "hidden_dims": [256, 128, 64, 32],
        "output_dim": env_config["wrapper"]["num_actions"],
        "activation": "leaky_relu",
        "softmax": False,
        "batch_norm": True,
        "skip_connection": False,
        "head": True,
        "dropout_p": 0.0
    },
    "optimization": {
        "optim_cls": "rmsprop",
        "optim_params": {"lr": 0.05}
    }
}

dqn_config = {
    "reward_discount": .0,
    "update_target_every": 5,
    "num_epochs": 10,
    "soft_update_coeff": 0.1,
    "double": False,
    "replay_memory_capacity": 10000,
    "random_overwrite": False,
    "prioritized_replay_kwargs": {
        "batch_size": 32,
        "alpha": 0.6,
        "beta": 0.4,
        "beta_step": 0.001,
        "max_priority": 1e8
    }
}

exploration_config = {
    "last_ep": 10,
    "initial_value": 0.4,
    "final_value": 0.0,
    "splits": [(5, 0.32)]
}


class QNet(DiscreteQNet):
    def __init__(self, component: nn.Module, optim_option: OptimOption=None, device=None):
        super().__init__(component, optim_option=optim_option, device=device)

    def forward(self, states):
        states = torch.from_numpy(np.asarray(states)).to(self.device)
        if len(states.shape) == 1:
            states = states.unsqueeze(dim=0)
        return self.component(states)


def get_dqn_policy(name: str, remote: bool = False):
    qnet = QNet(
        FullyConnectedBlock(**model_config["network"]),
        optim_option=OptimOption(**model_config["optimization"])
    )
    exploration = EpsilonGreedyExploration()
    exploration.register_schedule(
        scheduler_cls=MultiPhaseLinearExplorationScheduler,
        param_name="epsilon",
        **exploration_config
    )
    return DQN(name, qnet, exploration=exploration, **dqn_config, remote=remote)
