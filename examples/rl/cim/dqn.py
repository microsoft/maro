# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

import numpy as np
import torch
import torch.nn as nn

from maro.rl.algorithms import DQN
from maro.rl.experience import ExperienceStore, UniformSampler
from maro.rl.exploration import EpsilonGreedyExploration, MultiPhaseLinearExplorationScheduler
from maro.rl.model import DiscreteQNet, FullyConnectedBlock, OptimOption
from maro.rl.policy import CorePolicy


cim_path = os.path.dirname(os.path.realpath(__file__))
if cim_path not in sys.path:
    sys.path.insert(0, cim_path)
from env_wrapper import STATE_DIM, env_config

q_net_config = {
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
    "train_epochs": 10,
    "soft_update_coeff": 0.1,
    "double": False
}


exploration_config = {
    "last_ep": 10,
    "initial_value": 0.4,
    "final_value": 0.0,
    "splits": [(5, 0.32)]
}


experience_config = {
    "memory": {
        "rollout": {"capacity": 1000, "overwrite_type": "rolling"},
        "update": {"capacity": 100000, "overwrite_type": "rolling"}
    },
    "sampler": {
        "rollout": {
            "batch_size": -1,
            "replace": False
        },
        "update": {
            "batch_size": 128,
            "replace": True
        }
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
        FullyConnectedBlock(**q_net_config["network"]),
        optim_option=OptimOption(**q_net_config["optimization"]) if learning else None
    )
    exploration = EpsilonGreedyExploration()
    exploration.register_schedule(
        scheduler_cls=MultiPhaseLinearExplorationScheduler,
        param_name="epsilon",
        **exploration_config
    )
    return CorePolicy(
        DQN(qnet, **dqn_config, exploration=exploration),
        ExperienceStore(**experience_config)
    )
    if mode == "update":
        exp_memory = ExperienceStore(**config["experience_store"]["update"])
        exp_sampler_kwargs = config["sampler"]["update"]
    else:
        exp_memory = ExperienceStore(**config["experience_store"]["rollout" if mode == "inference" else "update"])
        exp_sampler_kwargs = config["sampler"]["rollout" if mode == "inference" else "update"]

    algorithm = 
    return CorePolicy(algorithm, exp_memory, exp_sampler_cls=UniformSampler, exp_sampler_kwargs=exp_sampler_kwargs)
