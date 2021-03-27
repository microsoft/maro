# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from torch import nn
from torch.optim import RMSprop

from maro.rl import OptimOption, OverwriteType

from examples.cim.common import common_config

input_dim = (
    (common_config["look_back"] + 1) *
    (common_config["max_ports_downstream"] + 1) *
    len(common_config["port_attributes"]) +
    len(common_config["vessel_attributes"])
)

config = {
    "agent": {
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
    },
    "training": {
        "env": {
            "scenario": "cim",
            "topology": "toy.4p_ssdd_l0.0",
            "durations": 1120,
        },
        "max_episode": 100,
        "min_experiences_to_train": 1024,
        "train_iter": 10,
        "batch_size": 128,
        "replay_memory": {
            "replay_memory_size": 2000,
            "replay_memory_overwrite_type": OverwriteType.RANDOM,
        },
        # "prioritized_sampling_by_loss": True,
        "exploration": {
            "parameter_names": ["epsilon"],
            "split": 0.5,
            "start": 0.4,
            "mid": 0.32,
            "end": 0.0
        },
    },
    "distributed": {
        "group": "cim-dqn",
        "num_actors": 2,
        "redis_host": "localhost",
        "redis_port": 6379,
        "learner_update_trigger": 2,
        "replay_sync_interval": 100
    }
}
