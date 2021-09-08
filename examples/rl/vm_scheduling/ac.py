# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

import numpy as np
import torch

from maro.rl.experience import ReplayMemory, UniformSampler
from maro.rl.model import DiscreteACNet, FullyConnected, OptimOption
from maro.rl.policy.algorithms import ActorCritic, ActorCriticConfig

vm_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, vm_path)
from env_wrapper import NUM_PMS, STATE_DIM

config = {
    "model": {
        "network": {
            "actor": {
                "input_dim": STATE_DIM,
                "output_dim": NUM_PMS + 1,  # action could be any PM or postponement, hence the plus 1
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
        "critic_loss_cls": "mse",
        "critic_loss_coeff": 0.1
    },
    "replay_memory": {
        "rollout": {"capacity": 10000, "overwrite_type": "rolling"},
        "update": {"capacity": 50000, "overwrite_type": "rolling"}
    },
    "sampler": {
        "rollout": {"batch_size": -1, "replace": False},
        "update": {"batch_size": 128, "replace": True}
    }
}


def get_ac_policy(mode="update"):
    class MyACNet(DiscreteACNet):
        def forward(self, states, actor: bool = True, critic: bool = True):
            if isinstance(states, dict):
                states = [states]
            inputs = torch.from_numpy(np.asarray([st["model"] for st in states])).to(self.device)
            masks = torch.from_numpy(np.asarray([st["mask"] for st in states])).to(self.device)
            if len(inputs.shape) == 1:
                inputs = inputs.unsqueeze(dim=0)
            return (
                self.component["actor"](inputs) * masks if actor else None,
                self.component["critic"](inputs) if critic else None
            )

    ac_net = MyACNet(
        component={
            "actor": FullyConnected(**config["model"]["network"]["actor"]),
            "critic": FullyConnected(**config["model"]["network"]["critic"])
        },
        optim_option={
            "actor":  OptimOption(**config["model"]["optimization"]["actor"]),
            "critic": OptimOption(**config["model"]["optimization"]["critic"])
        } if mode != "inference" else None
    )
    if mode == "update":
        exp_store = ReplayMemory(**config["replay_memory"]["update"])
        exp_sampler_kwargs = config["sampler"]["update"]
    else:
        exp_store = ReplayMemory(**config["replay_memory"]["rollout" if mode == "inference" else "update"])
        exp_sampler_kwargs = config["sampler"]["rollout" if mode == "inference" else "update"]

    return ActorCritic(
        ac_net, ActorCriticConfig(**config["algorithm"]), exp_store,
        experience_sampler_cls=UniformSampler,
        experience_sampler_kwargs=exp_sampler_kwargs
    )
