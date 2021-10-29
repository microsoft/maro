# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import numpy as np


env_config = {
    "topology": "building121",
    "durations": 4630
}

training_config = {
    "num_episodes": 500,
    "evaluate_interval": 10,
    "checkpoint_path": "/home/Jinyu/maro/examples/hvac/ddpg/checkpoints",
    "log_path": "/home/Jinyu/maro/examples/hvac/ddpg/logs",
}

state_config = {
    "attributes": ["kw", "at", "dat", "mat"]
}

ac_net_config = {
    "input_dim": len(state_config["attributes"]),
    "output_dim": 2,
    "output_lower_bound": [0.6, 40],    # Action lower bound
    "output_upper_bound": [1.3, 70],    # Action upper bound
    "actor_hidden_dims": [128, 128, 64],
    "critic_hidden_dims": [128, 128, 64],
    "actor_activation": torch.nn.ReLU,
    "critic_activation": torch.nn.ReLU,
    "actor_optimizer": torch.optim.Adam,
    "critic_optimizer": torch.optim.RMSprop,
    "actor_lr": 0.01,
    "critic_lr": 0.01
}

