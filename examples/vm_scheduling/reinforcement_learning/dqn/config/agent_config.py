# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from torch import nn

from maro.rl import OptimOption

from .training_config import training_config
from agent import CombineNet, SequenceNet

input_dim = (
    (training_config["env"]["wrapper"]["pm_num"] * 2)
    * training_config["env"]["wrapper"]["pm_window_size"]
    + (5 * training_config["env"]["wrapper"]["vm_window_size"])
)

agent_config = {
    "model_type": CombineNet,
    "model": {
        "input_dim": input_dim,
        "output_dim": training_config["env"]["wrapper"]["pm_num"] + 1,   # number of possible actions
        "hidden_dims": [64, 128, 256],
        "activation": nn.LeakyReLU,
        "softmax": False,
        "batch_norm": False,
        "skip_connection": False,
        "head": True,
        "dropout_p": 0.0
    },
    "rule_agent": {
        "pm_num": training_config["env"]["wrapper"]["pm_num"],
        "algorithm": {
            "type": "examples.vm_scheduling.rule_based_algorithm.best_fit.BestFit",
            "args": {
                "metric_type": "remaining_cpu_cores"
            }
        }
    },
    "optimization": OptimOption(
        optim_cls="sgd",
        optim_params={"lr": 0.0005},
        scheduler_cls="cosine_annealing_warm_restarts",
        scheduler_params={"T_0": 500, "T_mult": 2}
    ),
    "experience_manager": {
        "capacity": 10000,
        "overwrite_type": "rolling",
        "batch_size": 256
    },
    "algorithm_config": {
        "reward_discount": 0.9,
        "target_update_freq": 5,
        "train_epochs": 100,
        "gradient_iters": 1,
        "soft_update_coefficient": 0.1,
        "double": True,
        "loss_cls": nn.SmoothL1Loss
    }
}
