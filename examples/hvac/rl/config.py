# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

algorithm = "ddpg"
experiment_name = f"{algorithm}_test"

env_config = {
    "topology": "building121",
    "durations": 500
}

training_config = {
    # Test
    "test": False,
    # "model_path": "/home/Jinyu/maro/examples/hvac/rl/checkpoints/2021-11-03 04:33:20 ddpg_rewrite_Bonsai_env_positive/ddpg_49",
    "model_path": "/home/Jinyu/maro/examples/hvac/rl/checkpoints/2021-11-03 04:32:29 ddpg_rewrite_V2_env_positive/ddpg_49",
    # Train
    "load_model": False,
    "num_episodes": 200,
    "evaluate_interval": 10,
    "checkpoint_path": os.path.join(CURRENT_DIR, "checkpoints"),
    "log_path": os.path.join(CURRENT_DIR, "logs"),
}

state_config = {
    "attributes": ["kw", "at", "dat", "mat"],
    # "attributes": ["kw", "at", "mat"],   # The one for Bonsai
    "normalize": True,
}

reward_config = {
    # Bonsai
    "type": "Bonsai",  # Bonsai, V2, V3, V4, V5
    # V2
    "V2_efficiency_factor": 10,
    "V2_das_diff_factor": -2,
    "V2_sps_diff_factor": 0,
    "V2_constraints_factor": -0.5,
    "V2_lower_bound": None, # -2.5
    # V3
    "V3_threshold": -5,
    # V4
    "V4_kw_factor": 1,
    "V4_das_diff_penalty_factor": -0.05,
    "V4_dat_penalty_factor": -0.2,
    # V5
    "V5_kw_factor": 4,
    "V5_dat_penalty_factor": -0.06,
}

state_dim = len(state_config["attributes"])
action_dim = 2
action_lower_bound = [0.6, 53]
action_upper_bound = [1.1, 65]

############################################## POLICIES ###############################################

#### DDPG

ac_net_config = {
    "input_dim": state_dim,
    "output_dim": action_dim,
    "output_lower_bound": action_lower_bound,    # Action lower bound, the one for Bonsai
    "output_upper_bound": action_upper_bound,    # Action upper bound, the one for Bonsai
    "actor_hidden_dims": [256, 256],
    "critic_hidden_dims": [256, 256],
    "actor_activation": torch.nn.Tanh,
    "critic_activation": torch.nn.Tanh,
    "actor_optimizer": torch.optim.Adam,
    "critic_optimizer": torch.optim.RMSprop,
    "actor_lr": 0.001,
    "critic_lr": 0.001
}


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


####  SAC

sac_policy_net_config = {
    "input_dim": state_dim,
    "hidden_dims": [256, 256],
    "output_dim": 64,
    "activation": torch.nn.Tanh,
    "softmax": False,
    "batch_norm": True,
    "skip_connection": False,
    "head": True,
    "dropout_p": 0.0
}


sac_policy_net_optim_config = (torch.optim.Adam, {"lr": 0.001})

sac_q_net_config = {
    "input_dim": state_dim + action_dim,
    "hidden_dims": [256, 256],
    "output_dim": 1,
    "activation": torch.nn.Tanh,
    "softmax": False,
    "batch_norm": False,
    "skip_connection": False,
    "head": True,
    "dropout_p": 0.0
}

sac_q_net_optim_config = (torch.optim.Adam, {"lr": 0.001})

sac_config = {
    "reward_discount": 0.99,
    "soft_update_coeff": 0.9,
    "alpha": 0.2,
    "replay_memory_capacity": 10000,
    "random_overwrite": False,
    "warmup": 5000,
    "update_target_every": 5,
    "rollout_batch_size": 256,
    "train_batch_size": 256
}
