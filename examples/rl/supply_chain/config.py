# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

env_conf = {
    "scenario": "supply_chain",
    # Currently available topologies are "sample" or "random". New topologies must consist of a single folder
    # that contains a single config.yml and should be placed under /maro/simulator/scenarios/supply_chain/topologies
    "topology": "random_25",
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
    "device": "cpu",
    "network": {
        # "input_dim" will be filled by env_info.py
        "hidden_dims": [256, 128, 32],
        "output_dim": NUM_CONSUMER_ACTIONS,
        "activation": "leaky_relu",  # refer to maro/maro/rl/utils/torch_cls_index.py for the mapping of strings to torch activation classes.
        "softmax": True,
        "batch_norm": False,
        "skip_connection": False,
        "head": True,
        "dropout_p": 0.0
    },
    "optimization": {
        "optim_cls": "adam",  # refer to maro/maro/rl/utils/torch_cls_index.py for the mapping of strings to torch optimizer classes.
        "optim_params": {"lr": 0.0005}
    }
}

dqn_conf = {    
    "reward_discount": .99,
    "num_epochs": 10,
    "update_target_every": 4,   # How many training iteration, to update DQN target model
    "soft_update_coeff": 0.01,
    "double": True,   # whether to enable double DQN
    "replay_memory_capacity": 10000,
    "random_overwrite": False,
    "rollout_batch_size": 2560,
    "train_batch_size": 256,
}

exploration_conf = {
    "last_ep": 10,
    "initial_value": 0.8,   # Here (start: 0.4, end: 0.0) means: the exploration rate will start at 0.4 and decrease linearly to 0.0 in the last episode.
    "final_value": 0.0
}
