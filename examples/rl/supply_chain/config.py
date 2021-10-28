# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

from maro.rl.exploration import LinearExplorationScheduler, epsilon_greedy

env_conf = {
    "scenario": "supply_chain",
    # Currently available topologies are "sample" or "random". New topologies must consist of a single folder
    # that contains a single config.yml and should be placed under /maro/simulator/scenarios/supply_chain/topologies
    "topology": "random_50",
    "durations": 100  # number of ticks per episode
}

keys_in_state = [
    (None, ['is_over_stock', 'is_out_of_stock', 'is_below_rop', 'consumption_hist']),
    ('storage_capacity', ['storage_utilization']),
    ('sale_mean', [
        'sale_std',
        'sale_hist',
        'pending_order',
        'inventory_in_stock',
        'inventory_in_transit',
        'inventory_estimated',
        'inventory_rop'
    ]),
    ('max_price', ['sku_price', 'sku_cost'])
]

# Sku related agent types
sku_agent_types = {"consumer", "consumerstore", "producer", "product", "productstore"}
distribution_features = ("remaining_order_quantity", "remaining_order_number")
seller_features = ("total_demand", "sold", "demand")

NUM_CONSUMER_ACTIONS = 10
NUM_RL_POLICIES = 5

q_net_conf = {  
    # "input_dim" will be filled by env_info.py
    "hidden_dims": [256, 128, 32],
    "output_dim": NUM_CONSUMER_ACTIONS,
    "activation": torch.nn.LeakyReLU,
    "softmax": True,
    "batch_norm": False,
    "skip_connection": False,
    "head": True,
    "dropout_p": 0.0
}

q_net_optim_conf = (torch.optim.Adam, {"lr": 0.0005})

dqn_conf = {    
    "reward_discount": .99,
    "num_epochs": 10,
    "update_target_every": 4,   # How many training iteration, to update DQN target model
    "soft_update_coeff": 0.01,
    "double": True,   # whether to enable double DQN
    "exploration_strategy": (epsilon_greedy, {"epsilon": 0.4}),
    "exploration_scheduling_options": [(
        "epsilon", LinearExplorationScheduler, {
            "last_ep": 10,
            "initial_value": 0.8,
            "final_value": 0.0,
        }
    )],
    "replay_memory_capacity": 10000,
    "random_overwrite": False,
    "warmup": 1000,
    "rollout_batch_size": 2560,
    "train_batch_size": 256,
    "device": "cpu"
}
