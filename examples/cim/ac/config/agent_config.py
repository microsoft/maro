# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from torch import nn
from torch.optim import Adam, RMSprop

from maro.rl import OptimOption

from examples.cim.common import PORT_ATTRIBUTES, VESSEL_ATTRIBUTES, ACTION_SPACE, LOOK_BACK, MAX_PORTS_DOWNSTREAM

input_dim = (LOOK_BACK + 1) * (MAX_PORTS_DOWNSTREAM + 1) * len(PORT_ATTRIBUTES) + len(VESSEL_ATTRIBUTES)
agent_config = {
    "model": {
        "actor": {
            "input_dim": input_dim,
            "output_dim": len(ACTION_SPACE),
            "hidden_dims": [256, 128, 64],
            "activation": nn.Tanh,
            "softmax": True,
            "batch_norm": False,
            "head": True
        },
        "critic": {
            "input_dim": input_dim,
            "output_dim": 1,
            "hidden_dims": [256, 128, 64],
            "activation": nn.LeakyReLU,
            "softmax": False,
            "batch_norm": True,
            "head": True
        }
    },
    "optimization": {
        "actor": OptimOption(optim_cls=Adam, optim_params={"lr": 0.001}),
        "critic": OptimOption(optim_cls=RMSprop, optim_params={"lr": 0.001})
    },
    "hyper_params": {
        "reward_discount": .0,
        "critic_loss_func": nn.SmoothL1Loss(),
        "train_iters": 10,
        "actor_loss_coefficient": 0.1,
        "k": 1,
        "lam": 0.0
        # "clip_ratio": 0.8
    }
}
