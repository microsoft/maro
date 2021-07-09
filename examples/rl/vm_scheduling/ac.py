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
    }
}


class MyACNet(DiscreteACNet):
    def forward(self, states, actor: bool = True, critic: bool = True):
        inputs = torch.from_numpy(np.asarray([st["model"] for st in states])).to(self.device)
        
        if len(states.shape) == 1:
            states = states.unsqueeze(dim=0)
        return (
            self.component["actor"](inputs) if actor else None,
            self.component["critic"](inputs) if critic else None
        )

    def get_action(self, states, training=True):
        """
        Given Q-values for a batch of states, return the action index and the corresponding maximum Q-value
        for each state.
        """
        states, legal_action = states
        legal_action = torch.from_numpy(np.asarray(legal_action)).to(self.device)

        if not training:
            action_prob = self.forward(states, critic=False)[0]
            _, action = (action_prob + (legal_action - 1) * 1e8).max(dim=1)
            return action, action_prob

        action_prob = Categorical(self.forward(states, critic=False)[0] * legal_action)  # (batch_size, action_space_size)
        action = action_prob.sample()
        log_p = action_prob.log_prob(action)

        return action, log_p


def get_ac_policy():
    ac_net = MyACNet(
        component={
            "actor": config["actor_type"](**config["model"]["network"]["actor"]),
            "critic": agent_config["critic_type"](**config["model"]["network"]["critic"])
        },
        optim_option={
            "actor":  OptimOption(**config["model"]["optimization"]["actor"]),
            "critic": OptimOption(**config["model"]["optimization"]["critic"])
        }
    )
    experience_manager = ExperienceManager(**config["experience_manager"])
    return ActorCritic(ac_net, experience_manager, ActorCriticConfig(**config["algorithm_config"]))
