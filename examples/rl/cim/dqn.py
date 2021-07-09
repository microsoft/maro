# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from maro.rl.experience import ExperienceManager
from maro.rl.exploration import EpsilonGreedyExploration, MultiPhaseLinearExplorationScheduler
from maro.rl.model import DiscreteQNet, FullyConnectedBlock, OptimOption
from maro.rl.policy.algorithms import DQN, DQNConfig

cim_path = os.path.dirname(os.path.realpath(__file__))
if cim_path not in sys.path:
    sys.path.insert(0, cim_path)
from env_wrapper import STATE_DIM, env_config

config = {
    "model": {
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
    },
    "algorithm": {
        "reward_discount": .0,
        "target_update_freq": 5,
        "train_epochs": 10,
        "soft_update_coefficient": 0.1,
        "double": False
    },
    "experience_manager": {
        "rollout": {      # for experience managers in actor processes
            "capacity": 1000,
            "overwrite_type": "rolling",
            "batch_size": -1,
            "replace": False
        },
        "learning": {      # for experience managers in the learner process
            "capacity": 100000,
            "overwrite_type": "rolling",
            "batch_size": 128,
            "alpha": 0.6,
            "beta": 0.4,
            "beta_step": 0.001
        }
    },
    "exploration": {
        "last_ep": 10,
        "initial_value": 0.4,
        "final_value": 0.0,
        "splits": [(5, 0.32)]
    }
}


class QNet(DiscreteQNet):
    def __init__(self, component: nn.Module, optim_option: OptimOption=None, device=None):
        super().__init__(component, optim_option=optim_option, device=device)

    def forward(self, states):
        states = torch.from_numpy(np.asarray(states)).to(self.device)
        if len(states.shape) == 1:
            states = states.unsqueeze(dim=0)
        return self.component(states)


def get_dqn_policy(learning: bool = True):
    qnet = QNet(
        FullyConnectedBlock(**config["model"]["network"]),
        optim_option=OptimOption(**config["model"]["optimization"]) if learning else None
    )
    if learning:
        exp_manager = ExperienceManager(**config["experience_manager"]["learning"])
        exploration = None
    else:
        exp_manager = ExperienceManager(**config["experience_manager"]["rollout"])
        exploration = EpsilonGreedyExploration()
        print("expl ID: ", id(exploration))
        exploration.register_schedule(
            scheduler_cls=MultiPhaseLinearExplorationScheduler,
            param_name="epsilon",
            **config["exploration"]
        )

    return DQN(qnet, exp_manager, DQNConfig(**config["algorithm"]), exploration=exploration)
