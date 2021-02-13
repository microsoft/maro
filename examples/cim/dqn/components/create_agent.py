# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch.nn as nn
from torch.optim import RMSprop

from maro.rl import DQN, DQNConfig, FullyConnectedBlock, OptimOption, SimpleMultiHeadModel


def create_dqn_agent(config):
    q_net = FullyConnectedBlock(
        activation=nn.LeakyReLU,
        is_head=True,
        **config.model
    )            
    q_model = SimpleMultiHeadModel(
        q_net,
        optim_option=OptimOption(optim_cls=RMSprop, optim_params=config.optimizer)
    )
    return DQN(q_model, DQNConfig(**config.hyper_params, loss_cls=nn.SmoothL1Loss))
