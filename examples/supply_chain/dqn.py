# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch

from maro.rl import DQN, DQNConfig, FullyConnectedBlock, OptimOption, DiscreteQNet, ExperienceManager

config = {
    "model": {   # Edit the get_dqn_agent() code in examples\supply_chain\agent.py if you need to customize the model.
        "device": "cpu",
        "network": {
            "hidden_dims": [256, 128, 32],
            "output_dim": 10,
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
    },
    "algorithm": {
        "reward_discount": .99,
        "train_epochs": 10,
        "target_update_freq": 4,   # How many training iteration, to update DQN target model
        "soft_update_coefficient": 0.01,
        "double": True   # whether to enable double DQN
    },
    "experience_manager": {
        "rollout": {      # for experience managers in actor processes
            "capacity": 1000,
            # This determines how existing experiences are replaced when adding new experiences to a full experience
            # memory. Must be one of "rolling" or "random". If "rolling", experiences will be replaced sequentially,
            # with the oldest one being the first to be replaced. If "random", experiences will be replaced randomly.
            "overwrite_type": "rolling",  
            "batch_size": 128,
            "replace": False
        },
        "training": {      # for experience managers in the learner process
            "capacity": 128000,
            "overwrite_type": "rolling",
            "batch_size": 2560,
            "replace": True
        }
    },
    "update_trigger": 16,
    "warmup": 1  
}


class QNet(DiscreteQNet):
    def forward(self, states):
        states = torch.from_numpy(np.asarray(states)).to(self.device)
        if len(states.shape) == 1:
            states = states.unsqueeze(dim=0)
        return self.component(states)


def get_dqn_policy_for_rollout():
    qnet = QNet(FullyConnectedBlock(**config["model"]["network"]))
    return DQN(
        qnet,
        ExperienceManager(**config["experience_manager"]["rollout"]),
        DQNConfig(**config["algorithm"]),
        update_trigger=1e8  # set to a large number to ensure that the roll-out workers don't update policies
    )


def get_dqn_policy_for_training():
    qnet = QNet(
        FullyConnectedBlock(**config["model"]["network"]),
        optim_option=OptimOption(**config["model"]["optimization"])
    )
    return DQN(
        qnet,
        ExperienceManager(**config["experience_manager"]["training"]),
        DQNConfig(**config["algorithm"]),
        update_trigger=config["update_trigger"],
        warmup=config["warmup"]
    )
