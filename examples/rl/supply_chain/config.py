# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

q_net_conf = {  
    "device": "cpu",
    "network": {
        "input_dim": STATE_DIM,
        "hidden_dims": [256, 128, 32],
        "output_dim": NUM_ACTIONS,
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
    "train_epochs": 10,
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

