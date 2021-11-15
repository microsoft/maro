# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch.optim import Adam, RMSprop

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

# ############################################# POLICIES ###############################################

algorithm = "maac"

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

critic_conf = {
    "state_dim": state_dim,
    "action_dims": [1]
}

critic_net_conf = {
    "input_dim": critic_conf["state_dim"] + sum(critic_conf["action_dims"]),
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
    "num_epoch": 10
    # "entropy_coef": 0.01,
    # "clip_ratio": 0.8   # for PPO
}
