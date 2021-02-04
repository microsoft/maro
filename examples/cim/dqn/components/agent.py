# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch.nn as nn
from torch.optim import RMSprop

from maro.rl import DQN, DQNConfig, FullyConnectedBlock, OptimOption, SimpleMultiHeadModel
from maro.utils import set_seeds


def create_dqn_agents(agent_names, config):
    set_seeds(config.seed)
    agent_dict = {}
    for name in agent_names:
        q_net = FullyConnectedBlock(
            activation=nn.LeakyReLU,
            is_head=True,
            **config.model
        )            
        learning_model = SimpleMultiHeadModel(
            q_net,
            optim_option=OptimOption(optim_cls=RMSprop, optim_params=config.optimizer)
        )
        agent_dict[name] = DQN(
            name, learning_model, DQNConfig(**config.hyper_params, loss_cls=nn.SmoothL1Loss)
        )

    return agent_dict
