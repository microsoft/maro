# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

import numpy as np
import torch

from maro.rl.algorithms import ActorCritic
from maro.rl.experience import ExperienceStore
from maro.rl.model import DiscreteACNet, FullyConnectedBlock, OptimOption
from maro.rl.policy import CorePolicy


cim_path = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, cim_path)
from env_wrapper import STATE_DIM, env_config

ac_net_config = {
    "network": {
        "actor": {
            "input_dim": STATE_DIM,
            "hidden_dims": [256, 128, 64],
            "output_dim": env_config["wrapper"]["num_actions"],
            "activation": "tanh",
            "softmax": True,
            "batch_norm": False,
            "head": True
        },
        "critic": {
            "input_dim": STATE_DIM,
            "hidden_dims": [256, 128, 64],
            "output_dim": 1,
            "activation": "leaky_relu",
            "softmax": False,
            "batch_norm": True,
            "head": True
        }
    },
    "optimization": {
        "actor": {
            "optim_cls": "adam",
            "optim_params": {"lr": 0.001}
        },
        "critic": {
            "optim_cls": "rmsprop",
            "optim_params": {"lr": 0.001}
        }
    }
}

ac_config = {
    "reward_discount": .0,
    "critic_loss_cls": "smooth_l1",
    "critic_loss_coeff": 0.1,
    "entropy_coeff": 0.01,
    # "clip_ratio": 0.8   # for PPO
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


def get_ac_policy(learning: bool = True):
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

    ac_net = MyACNET(
        component={
            "actor": FullyConnectedBlock(**ac_net_config["network"]["actor"]),
            "critic": FullyConnectedBlock(**ac_net_config["network"]["critic"])
        },
        optim_option={
            "actor":  OptimOption(**ac_net_config["optimization"]["actor"]),
            "critic": OptimOption(**ac_net_config["model"]["optimization"]["critic"])
        } if learning else None
    )

    return CorePolicy(
        ActorCritic(ac_net, ac_config["algorithm"]),
        ExperienceStore(**experience_config["memory"])
    )
