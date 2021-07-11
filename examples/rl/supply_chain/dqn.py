# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

import numpy as np
import torch

from maro.rl.experience import ExperienceManager
from maro.rl.exploration import EpsilonGreedyExploration, LinearExplorationScheduler
from maro.rl.model import FullyConnectedBlock, OptimOption, DiscreteQNet
from maro.rl.policy.algorithms import DQN, DQNConfig

sc_path = os.path.dirname(os.path.realpath(__file__))
if sc_path not in sys.path:
    sys.path.insert(0, sc_path)
from env_wrapper import NUM_ACTIONS, STATE_DIM

config = {
    "model": {   # Edit the get_dqn_agent() code in examples\supply_chain\agent.py if you need to customize the model.
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
            "capacity": 10000,
            # This determines how existing experiences are replaced when adding new experiences to a full experience
            # memory. Must be one of "rolling" or "random". If "rolling", experiences will be replaced sequentially,
            # with the oldest one being the first to be replaced. If "random", experiences will be replaced randomly.
            "overwrite_type": "rolling",  
            "batch_size": 128,
            "replace": False
        },
        "training": {      # for experience managers in the learner process
            "capacity": 100000,
            "overwrite_type": "rolling",
            "batch_size": 256,
            "replace": True
        }
    },
    "exploration_config": {
        "last_ep": 10,
        "initial_value": 0.8,   # Here (start: 0.4, end: 0.0) means: the exploration rate will start at 0.4 and decrease linearly to 0.0 in the last episode.
        "final_value": 0.0
    }
}


class QNet(DiscreteQNet):
    def forward(self, states):
        states = torch.from_numpy(np.asarray(states)).to(self.device)
        if len(states.shape) == 1:
            states = states.unsqueeze(dim=0)
        return self.component(states)


def get_dqn_policy(learning=True):
    qnet = QNet(
        FullyConnectedBlock(**config["model"]["network"]),
        optim_option=OptimOption(**config["model"]["optimization"]) if learning else None
    )
    if learning:
        exp_manager = ExperienceManager(**config["experience_manager"]["learning"])
        exploration = None
    else:
        exp_manager = ExperienceManager(**config["experience_manager"]["rollout"])
        exploration = EpsilonGreedyExploration()
        exploration.register_schedule(
            scheduler_cls=LinearExplorationScheduler,
            param_name="epsilon",
            **config["exploration"]
        )

    return DQN(qnet, exp_manager, DQNConfig(**config["algorithm"]), exploration=exploration)
