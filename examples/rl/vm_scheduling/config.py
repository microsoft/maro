# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch
from torch.optim import Adam, SGD, lr_scheduler

from maro.rl.exploration import MultiLinearExplorationScheduler
from maro.simulator import Env


env_conf = {
    "scenario": "vm_scheduling",
    "topology": "azure.2019.10k",
    "start_tick": 0,
    "durations": 300,  # 8638
    "snapshot_resolution": 1
}

num_pms = Env(**env_conf).business_engine._pm_amount
pm_window_size = 1
num_features = 2 * num_pms * pm_window_size + 4

pm_attributes = ["cpu_cores_capacity", "memory_capacity", "cpu_cores_allocated", "memory_allocated"]
# vm_attributes = ["cpu_cores_requirement", "memory_requirement", "lifetime", "remain_time", "total_income"]


reward_shaping_conf = {
    "alpha": 0.0,
    "beta": 1.0
}
seed = 666

test_env_conf = {
    "scenario": "vm_scheduling",
    "topology": "azure.2019.10k.oversubscription",
    "start_tick": 0,
    "durations": 300,
    "snapshot_resolution": 1
}
test_reward_shaping_conf = {
    "alpha": 0.0,
    "beta": 1.0
}

test_seed = 1024

algorithm = "ac"  # "dqn" or "ac"

######################################### A2C settings ########################################
actor_net_conf = {
    "input_dim": num_features,
    "output_dim": num_pms + 1,  # action could be any PM or postponement, hence the plus 1
    "hidden_dims": [64, 32, 32],
    "activation": torch.nn.LeakyReLU,
    "softmax": True,
    "batch_norm": False,
    "head": True
}

critic_net_conf = {
    "input_dim": num_features,
    "output_dim": 1,
    "hidden_dims": [256, 128, 64],
    "activation": torch.nn.LeakyReLU,
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

######################################### DQN settings ########################################
q_net_conf = {
    "input_dim": num_features,
    "hidden_dims": [64, 128, 256],
    "output_dim": num_pms + 1,  # action could be any PM or postponement, hence the plus 1
    "activation": torch.nn.LeakyReLU,
    "softmax": False,
    "batch_norm": False,
    "skip_connection": False,
    "head": True,
    "dropout_p": 0.0
}

q_net_optim_conf = (SGD, {"lr": 0.0005})
q_net_lr_scheduler_conf = (lr_scheduler.CosineAnnealingWarmRestarts, {"T_0": 500, "T_mult": 2})


def masked_eps_greedy(states, actions, num_actions, *, epsilon):
    masks = states[:, num_features:]
    return np.array([
        action if np.random.random() > epsilon else np.random.choice(np.where(mask == 1)[0])
        for action, mask in zip(actions, masks)
    ])

dqn_conf = {
    "reward_discount": 0.9,
    "update_target_every": 5,
    "num_epochs": 100,
    "soft_update_coeff": 0.1,
    "double": False,
    "exploration_strategy": (masked_eps_greedy, {"epsilon": 0.4}),
    "exploration_scheduling_options": [(
        "epsilon", MultiLinearExplorationScheduler, {
            "splits": [(100, 0.32)],
            "initial_value": 0.4,
            "last_ep": 400,
            "final_value": 0.0,
        }
    )],
    "replay_memory_capacity": 10000,
    "rollout_batch_size": 2560,
    "train_batch_size": 256,
}
