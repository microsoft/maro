# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from torch import nn
from torch.optim import RMSprop

from maro.rl import DQN, DQNConfig, FullyConnectedBlock, OptimOption, PolicyGradient, SimpleMultiHeadModel

from examples.cim.common import PORT_ATTRIBUTES, VESSEL_ATTRIBUTES, ACTION_SPACE, LOOK_BACK, MAX_PORTS_DOWNSTREAM

agent_config = {
    "model": {
        "input_dim": (LOOK_BACK + 1) * (MAX_PORTS_DOWNSTREAM + 1) * len(PORT_ATTRIBUTES) + len(VESSEL_ATTRIBUTES),
        "output_dim": len(ACTION_SPACE),   # number of possible actions
        "hidden_dims": [256, 128, 64],
        "activation": nn.LeakyReLU,
        "softmax": False,
        "batch_norm": True,
        "skip_connection": False,
        "head": True,
        "dropout_p": 0.0
    },
    "optimization": OptimOption(optim_cls=RMSprop, optim_params={"lr": 0.05}),
    "hyper_params": {
        "reward_discount": .0,
        "loss_cls": nn.SmoothL1Loss,
        "target_update_freq": 5,
        "tau": 0.1,
        "double": False
    }
}
