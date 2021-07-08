# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

import numpy as np
import torch

from maro.rl.experience import ExperienceManager
from maro.rl.model import DiscreteACNet, FullyConnectedBlock, OptimOption
from maro.rl.policy.algorithms import ActorCritic, ActorCriticConfig

cim_path = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, cim_path)
from env_wrapper import STATE_DIM, env_config

config = {
    "model": {
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
                "output_dim": env_config["wrapper"]["num_actions"],
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
    },
    "algorithm": {
        "reward_discount": .0,
        "critic_loss_cls": "smooth_l1",
        "train_epochs": 10,
        "actor_loss_coefficient": 0.1,
        # "clip_ratio": 0.8   # for PPO
    },
    "experience_manager": {
        "rollout": {      # for experience managers in actor processes
            "capacity": 1000,
            "overwrite_type": "rolling",
            "batch_size": -1,
            "replace": False
        },
        "training": {      # for experience managers in the learner process
            "capacity": 100000,
            "overwrite_type": "rolling",
            "batch_size": 128,
            "alpha": 0.6,
            "beta": 0.4,
            "beta_step": 0.001
        }
    }    
}


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
            "actor": FullyConnectedBlock(**cfg["model"]["network"]["actor"]),
            "critic": FullyConnectedBlock(**cfg["model"]["network"]["critic"])
        },
        optim_option={
            "actor":  OptimOption(**cfg["model"]["optimization"]["actor"]),
            "critic": OptimOption(**cfg["model"]["optimization"]["critic"])
        }
    )
    experience_manager = ExperienceManager(**cfg["experience_manager"])
    return ActorCritic(name, ac_net, experience_manager, ActorCriticConfig(**cfg["algorithm_config"]))
