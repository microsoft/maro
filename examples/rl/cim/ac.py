# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

import numpy as np
import torch

from maro.rl.experience import ExperienceStore, UniformSampler
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
    },
    "algorithm": {
        "reward_discount": .0,
        "critic_loss_cls": "smooth_l1",
        "train_epochs": 10,
        "actor_loss_coefficient": 0.1,
        # "clip_ratio": 0.8   # for PPO
    },
    "experience_store": {
        "rollout": {"capacity": 1000, "overwrite_type": "rolling"},
        "update": {"capacity": 100000, "overwrite_type": "rolling"}
    },
    "sampler": {
        "rollout": {"batch_size": -1, "replace": False},
        "update": {"batch_size": 128, "replace": True}
    }
}


def get_ac_policy(mode="update"):
    assert mode in {"inference", "update", "inference-update"}
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
            "actor": FullyConnectedBlock(**config["model"]["network"]["actor"]),
            "critic": FullyConnectedBlock(**config["model"]["network"]["critic"])
        },
        optim_option={
            "actor":  OptimOption(**config["model"]["optimization"]["actor"]),
            "critic": OptimOption(**config["model"]["optimization"]["critic"])
        } if mode != "inference" else None
    )

    if mode == "update":
        exp_store = ExperienceStore(**config["experience_store"]["update"])
        experience_sampler_kwargs = config["sampler"]["update"]
    else:
        exp_store = ExperienceStore(**config["experience_store"]["rollout" if mode == "inference" else "update"])
        experience_sampler_kwargs = config["sampler"]["rollout" if mode == "inference" else "update"]

    return ActorCritic(
        ac_net, ActorCriticConfig(**config["algorithm"]), exp_store,
        experience_sampler_cls=UniformSampler,
        experience_sampler_kwargs=experience_sampler_kwargs
    )
