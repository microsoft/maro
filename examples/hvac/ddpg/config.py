# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import numpy as np

experiment_name = "test"

env_config = {
    "topology": "building121",
    "durations": 4629
}

training_config = {
    "test": False,
    "load_model": False,
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

#### DDPG

exploration_strategy = {
    "mean": 0,
    "stddev": 0.1,
    "min_action": ac_net_config["output_lower_bound"],
    "max_action": ac_net_config["output_upper_bound"],
}

exploration_mean_scheduler_options = {
    "start_ep": 0,
    "initial_value": exploration_strategy["stddev"],
    "splits": [(int(training_config["num_episodes"] * 0.6), exploration_strategy["stddev"])],
    "last_ep": training_config["num_episodes"] - 1,
    "final_value": 0,
}

ddpg_config = {
    "exploration_strategy": exploration_strategy,
    "exploration_mean_scheduler_options": exploration_mean_scheduler_options,
}

