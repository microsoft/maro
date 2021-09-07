# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch.optim import Adam, SGD


env_conf = {
    "scenario": "vm_scheduling",
    "topology": "azure.2019.10k",
    "start_tick": 0,
    "durations": 300,  # 8638
    "snapshot_resolution": 1
}

pm_attributes = ["cpu_cores_capacity", "memory_capacity", "cpu_cores_allocated", "memory_allocated"],
vm_attributes = ["cpu_cores_requirement", "memory_requirement", "lifetime", "remain_time", "total_income"],
        
shaping_conf = {
    "alpha": 0.0,
    "beta": 1.0,
    "pm_window_size": 1,
    "gamma": 0.9,
    "seed": 666
}


eval_env_conf = {
    "scenario": "vm_scheduling",
    "topology": "azure.2019.10k.oversubscription",
    "start_tick": 0,
    "durations": 300,
    "snapshot_resolution": 1
}

eval_shaping_conf = {
    "alpha": 0.0,
    "beta": 1.0,
    "pm_window_size": 1,
    "gamma": 0.9,
    "seed": 1024
}

actor_net_conf = {
    "input_dim": STATE_DIM,
    "output_dim": NUM_PMS + 1,  # action could be any PM or postponement, hence the plus 1
    "hidden_dims": [64, 32, 32],
    "activation": torch.nn.LeakyReLU,
    "softmax": True,
    "batch_norm": False,
    "head": True
}

critic_net_conf = {
    "input_dim": STATE_DIM,
    "output_dim": 1,
    "hidden_dims": [256, 128, 64],
    "activation": "leaky_relu",
    "softmax": False,
    "batch_norm": False,
    "head": True
}

actor_optim_conf = (Adam, {"lr": 0.0001})
critic_optim_conf = (SGD, {"lr": 0.001})

ac_conf = {
    "reward_discount": 0.9,
    "grad_iters": 100,
    "critic_loss_cls": torch.nn.MSELoss,
    "critic_loss_coeff": 0.1,
    "max_trajectory_len": 10000,
    "get_loss_on_rollout": False
}


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
    "experience_store": {
        "rollout": {"capacity": 10000, "overwrite_type": "rolling"},
        "update": {"capacity": 50000, "overwrite_type": "rolling"}
    },
    "sampler": {
        "rollout": {"batch_size": -1, "replace": False},
        "update": {"batch_size": 128, "replace": True}
    }
}

