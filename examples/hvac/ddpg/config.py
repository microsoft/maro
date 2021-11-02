# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

experiment_name = "ddpg_ori_reward"

env_config = {
    "topology": "building121",
    "durations": 4629
}

training_config = {
    "test": False,
    "load_model": False,
    "num_episodes": 200,
    "evaluate_interval": 10,
    "checkpoint_path": os.path.join(CURRENT_DIR, "checkpoints"),
    "log_path": os.path.join(CURRENT_DIR, "logs"),
}

state_config = {
    # "attributes": ["kw", "at", "dat", "mat"]
    "attributes": ["kw", "at", "mat"],   # The one for Bonsai
    "normalize": False,
}

reward_config = {
    "type": "Bonsai",  # Bonsai, V2, V3
    # Bonsai
    # V2
    "V2_efficiency_factor": 10,
    "V2_das_diff_factor": -2,
    "V2_sps_diff_factor": 0,
    "V2_constraints_factor": -0.5,
    "V2_lower_bound": None, # -2.5
    # V3
    "V3_threshold": -5,
}

state_dim = len(state_config["attributes"])
action_dim = 2
action_lower_bound = [0.6, 53]
action_upper_bound = [1.1, 65]

############################################## POLICIES ###############################################

algorithm = "ddpg"

#### DDPG

ac_net_config = {
    "input_dim": state_dim,
    "output_dim": action_dim,
    "output_lower_bound": action_lower_bound,    # Action lower bound, the one for Bonsai
    "output_upper_bound": action_upper_bound,    # Action upper bound, the one for Bonsai
    "actor_hidden_dims": [256, 256, 64],
    "critic_hidden_dims": [256, 256, 64],
    "actor_activation": torch.nn.Tanh,
    "critic_activation": torch.nn.Tanh,
    "actor_optimizer": torch.optim.Adam,
    "critic_optimizer": torch.optim.RMSprop,
    "actor_lr": 0.01,
    "critic_lr": 0.01
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
    "activation": torch.nn.LeakyReLU,
    "softmax": False,
    "batch_norm": True,
    "skip_connection": False,
    "head": True,
    "dropout_p": 0.0
}


sac_policy_net_optim_config = (torch.optim.Adam, {"lr": 0.01})

sac_q_net_config = {
    "input_dim": state_dim + action_dim,
    "hidden_dims": [256, 256, 64],
    "output_dim": 1,
    "activation": torch.nn.LeakyReLU,
    "softmax": False,
    "batch_norm": False,
    "skip_connection": False,
    "head": True,
    "dropout_p": 0.0
}

sac_q_net_optim_config = (torch.optim.Adam, {"lr": 0.01})

sac_config = {
    "reward_discount": 0.0,
    "soft_update_coeff": 0.1,
    "alpha": 0.2,
    "replay_memory_capacity": 10000,
    "random_overwrite": False,
    "warmup": 100,
    "update_target_every": 5,
    "rollout_batch_size": 128,
    "train_batch_size": 32
}
