# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch.optim import Adam, RMSprop

from maro.rl.exploration import MultiLinearExplorationScheduler, epsilon_greedy


env_conf = {
    "scenario": "cim",
    "topology": "toy.4p_ssdd_l0.0",
    "durations": 560
}

port_attributes = ["empty", "full", "on_shipper", "on_consignee", "booking", "shortage", "fulfillment"]
vessel_attributes = ["empty", "full", "remaining_space"]

state_shaping_conf = {
    "look_back": 7,
    "max_ports_downstream": 2
}

action_shaping_conf = {
    "action_space": [(i - 10) / 10 for i in range(21)],
    "finite_vessel_space": True,
    "has_early_discharge": True
}

reward_shaping_conf = {
    "time_window": 99,
    "fulfillment_factor": 1.0,
    "shortage_factor": 1.0,
    "time_decay": 0.97
}

# obtain state dimension from a temporary env_wrapper instance
state_dim = (
    (state_shaping_conf["look_back"] + 1) * (state_shaping_conf["max_ports_downstream"] + 1) * len(port_attributes)
    + len(vessel_attributes)
)

############################################## POLICIES ###############################################

algorithm = "maddpg"

# DQN settings
q_net_conf = {
    "input_dim": state_dim,
    "hidden_dims": [256, 128, 64, 32],
    "output_dim": len(action_shaping_conf["action_space"]),
    "activation": torch.nn.LeakyReLU,
    "softmax": False,
    "batch_norm": True,
    "skip_connection": False,
    "head": True,
    "dropout_p": 0.0
}

q_net_optim_conf = (RMSprop, {"lr": 0.05})

dqn_conf = {
    "reward_discount": .0,
    "update_target_every": 5,
    "num_epochs": 10,
    "soft_update_coef": 0.1,
    "double": False,
    "exploration_strategy": (epsilon_greedy, {"epsilon": 0.4}),
    "exploration_scheduling_options": [(
        "epsilon", MultiLinearExplorationScheduler, {
            "splits": [(2, 0.32)],
            "initial_value": 0.4,
            "last_ep": 5,
            "final_value": 0.0,
        }
    )],
    "replay_memory_capacity": 10000,
    "random_overwrite": False,
    "warmup": 100,
    "rollout_batch_size": 128,
    "train_batch_size": 32,
    # "prioritized_replay_kwargs": {
    #     "alpha": 0.6,
    #     "beta": 0.4,
    #     "beta_step": 0.001,
    #     "max_priority": 1e8
    # }
}


# AC settings
actor_net_conf = {
    "input_dim": state_dim,
    "hidden_dims": [256, 128, 64],
    "output_dim": len(action_shaping_conf["action_space"]),
    "activation": torch.nn.Tanh,
    "softmax": True,
    "batch_norm": False,
    "head": True
}

critic_net_conf = {
    "input_dim": state_dim,
    "hidden_dims": [256, 128, 64],
    "output_dim": 1,
    "activation": torch.nn.LeakyReLU,
    "softmax": False,
    "batch_norm": True,
    "head": True
}

actor_optim_conf = (Adam, {"lr": 0.001})
critic_optim_conf = (RMSprop, {"lr": 0.001})

ac_conf = {
    "reward_discount": .0,
    "grad_iters": 10,
    "critic_loss_cls": torch.nn.SmoothL1Loss,
    "min_logp": None,
    "critic_loss_coef": 0.1,
    "entropy_coef": 0.01,
    # "clip_ratio": 0.8   # for PPO
    "lam": .0,
    "get_loss_on_rollout": False
}

# MADDPG settings
num_agents = 4  # TODO: obtain num_agents in another way
action_dim = 1
centralized_critic_net_conf = {
    "input_dim": state_dim * num_agents + action_dim * num_agents,
    "hidden_dims": [256, 128, 64],
    "output_dim": 1,
    "activation": torch.nn.LeakyReLU,
    "softmax": False,
    "batch_norm": True,
    "head": True
}

centralized_critic_optim_conf = (RMSprop, {"lr": 0.001})

maddpg_conf = {
    "reward_discount": .0,
    "shared_state_dim": 0,
    "num_epochs": 10,
    "update_target_every": 5,
    "critic_loss_cls": torch.nn.SmoothL1Loss,
    "min_logp": None,
    "critic_loss_coef": 0.1,
    "soft_update_coef": 0.1,
    # "clip_ratio": 0.8   # for PPO
    "lam": .0,
    "replay_memory_capacity": 10000,
    "random_overwrite": False,
    "warmup": 100,
    "rollout_batch_size": 128,
    "train_batch_size": 32
}
