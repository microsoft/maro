# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

import numpy as np
import torch

from maro.rl.experience import ExperienceManager
from maro.rl.model import DiscreteACNet, FullyConnectedBlock, OptimOption
from maro.rl.policy.algorithms import ActorCritic, ActorCriticConfig

vm_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, vm_path)
from env_wrapper import STATE_DIM

config = {
    "model": {
        "network": {
            "actor": {
                "input_dim": STATE_DIM,
                "output_dim": 9,
                "hidden_dims": [64, 32, 32],
                "activation": "leaky_relu",
                "softmax": True,
                "batch_norm": False,
                "head": True
            },
            "critic": {
                "input_dim": STATE_DIM,
                "output_dim": 1,
                "hidden_dims": [256, 128, 64],
                "activation": "leaky_relu",
                "softmax": False,
                "batch_norm": False,
                "head": True
            }
        },
        "optimization": {
            "actor": {
                "optim_cls": "adam",
                "optim_params": {"lr": 0.0001}
            },
            "critic": {
                "optim_cls": "sgd",
                "optim_params": {"lr": 0.001}
            }
        }
    },
    "algorithm": {
        "reward_discount": 0.9,
        "train_epochs": 100,
        "gradient_iters": 1,
        "critic_loss_cls": "mse",
        "actor_loss_coefficient": 0.1
    },
    "experience_manager": {
        "capacity": 10000,
        "overwrite_type": "rolling",
        "batch_size": -1,
        "replace": False
    }
}


class MyACNet(DiscreteACNet):
    def forward(self, states, actor: bool = True, critic: bool = True):
        inputs = torch.from_numpy(np.asarray([st["model"] for st in states])).to(self.device)
        masks = torch.from_numpy(np.asarray([st["mask"] for st in states])).to(self.device)
        if len(inputs.shape) == 1:
            inputs = inputs.unsqueeze(dim=0)
        return (
            self.component["actor"](inputs) * masks if actor else None,
            self.component["critic"](inputs) if critic else None
        )


def get_ac_policy():
    ac_net = MyACNet(
        component={
            "actor": FullyConnectedBlock(**config["model"]["network"]["actor"]),
            "critic": FullyConnectedBlock(**config["model"]["network"]["critic"])
        },
        optim_option={
            "actor":  OptimOption(**config["model"]["optimization"]["actor"]),
            "critic": OptimOption(**config["model"]["optimization"]["critic"])
        }
    )
    experience_manager = ExperienceManager(**config["experience_manager"])
    return ActorCritic(ac_net, experience_manager, ActorCriticConfig(**config["algorithm"]))
