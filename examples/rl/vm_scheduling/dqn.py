  
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

import numpy as np
import torch

from maro.rl.experience import ReplayMemory, UniformSampler
from maro.rl.exploration import DiscreteSpaceExploration, MultiLinearExplorationScheduler
from maro.rl.model import DiscreteQNet, FullyConnected, OptimOption
from maro.rl.policy.algorithms import DQN, DQNConfig

vm_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, vm_path)
from env_wrapper import NUM_PMS, STATE_DIM

config = {
    "model": {
        "network": {
            "input_dim": STATE_DIM,
            "hidden_dims": [64, 128, 256],
            "output_dim": NUM_PMS + 1,  # action could be any PM or postponement, hence the plus 1
            "activation": "leaky_relu",
            "softmax": False,
            "batch_norm": False,
            "skip_connection": False,
            "head": True,
            "dropout_p": 0.0
        },
        "optimization": {
            "optim_cls": "sgd",
            "optim_params": {"lr": 0.0005},
            "scheduler_cls": "cosine_annealing_warm_restarts",
            "scheduler_params": {"T_0": 500, "T_mult": 2}
        }
    },
    "algorithm": {
        "reward_discount": 0.9,
        "update_target_every": 5,
        "train_epochs": 100,
        "soft_update_coeff": 0.1,
        "double": False
    },
    "replay_memory": {
        "rollout": {"capacity": 10000, "overwrite_type": "rolling"},
        "update": {"capacity": 50000, "overwrite_type": "rolling"}
    },
    "sampler": {
        "rollout": {"batch_size": -1, "replace": False},
        "update": {"batch_size": 256, "replace": True}
    },
    "exploration": {
        "last_ep": 400,
        "initial_value": 0.4,
        "final_value": 0.0,
        "splits": [(100, 0.32)]
    }
}


class MyQNet(DiscreteQNet):
    def __init__(self, component, optim_option, device: str = None):
        super().__init__(component, optim_option=optim_option, device=device)
        for mdl in self.modules():
            if isinstance(mdl, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(mdl.weight, gain=torch.nn.init.calculate_gain('leaky_relu'))

    def forward(self, states):
        if isinstance(states, dict):
            states = [states]
        inputs = torch.from_numpy(np.asarray([st["model"] for st in states])).to(self.device)
        masks = torch.from_numpy(np.asarray([st["mask"] for st in states])).to(self.device)
        if len(inputs.shape) == 1:
            inputs = inputs.unsqueeze(dim=0)
        q_for_all_actions = self.component(inputs)
        return q_for_all_actions + (masks - 1) * 1e8


class MaskedEpsilonGreedy(DiscreteSpaceExploration):
    def __init__(self, epsilon: float = .0):
        super().__init__()
        self.epsilon = epsilon

    def __call__(self, action, state):
        if isinstance(state, dict):
            state = [state]
        mask = [st["mask"] for st in state]
        return np.array([
            act if np.random.random() > self.epsilon else np.random.choice(np.where(mk == 1)[0])
            for act, mk in zip(action, mask)
        ])


def get_dqn_policy(mode="update"):
    assert mode in {"inference", "update", "inference-update"}
    q_net = MyQNet(
        FullyConnected(**config["model"]["network"]),
        optim_option=OptimOption(**config["model"]["optimization"]) if mode != "inference" else None
    )

    if mode == "update":
        exp_store = ReplayMemory(**config["replay_memory"]["update"])
        exploration = None
        exp_sampler_kwargs = config["sampler"]["update"]
    else:
        exp_store = ReplayMemory(**config["replay_memory"]["rollout"])
        exploration = MaskedEpsilonGreedy()
        exploration.register_schedule(
            scheduler_cls=MultiLinearExplorationScheduler,
            param_name="epsilon",
            **config["exploration"]
        )
        exp_store = ReplayMemory(**config["replay_memory"]["rollout" if mode == "inference" else "update"])
        exp_sampler_kwargs = config["sampler"]["rollout" if mode == "inference" else "update"]

    return DQN(
        q_net, DQNConfig(**config["algorithm"]), exp_store,
        experience_sampler_cls=UniformSampler,
        experience_sampler_kwargs=exp_sampler_kwargs,
        exploration=exploration
    )
