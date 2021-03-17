# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from torch import nn
from torch.optim import RMSprop

from maro.rl import DQN, DQNConfig, FullyConnectedBlock, OptimOption, PolicyGradient, SimpleMultiHeadModel

from examples.cim.common import common_config

input_dim = (
    (common_config["look_back"] + 1) *
    (common_config["max_ports_downstream"] + 1) *
    len(common_config["port_attributes"]) +
    len(common_config["vessel_attributes"])
)

agent_config = {
    "model": {
        "input_dim": input_dim,
        "output_dim": len(common_config["action_space"]),   # number of possible actions
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
