# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from torch import nn
from torch.optim import SGD, lr_scheduler

from maro.rl import OptimOption

from .training_config import training_config
from agent import CombineNet, SequenceNet

input_dim = (
    (training_config["env"]["wrapper"]["pm_num"] * 2)
    * training_config["env"]["wrapper"]["pm_window_size"]
    + (5 * training_config["env"]["wrapper"]["vm_window_size"])
)

agent_config = {
    "actor_type": CombineNet,
    "actor": {
        "input_dim": input_dim,
        "output_dim": training_config["env"]["wrapper"]["pm_num"] + 1,   # number of possible actions
        "hidden_dims": [64, 32, 32],
        "activation": "leaky_relu",
        "softmax": True,
        "batch_norm": False,
        "skip_connection": False,
        "head": True,
        "dropout_p": 0.0
    },
    "critic_type": CombineNet,
    "critic": {
        "input_dim": input_dim,
        "output_dim": 1,   # number of possible actions
        "hidden_dims": [256, 128, 64],
        "activation": "leaky_relu",
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
    "ilp_agent": {
        "pm_num": training_config["env"]["wrapper"]["pm_num"],
        "algorithm": {
            "ilp": {
                "solver": "CBC",          # GLPK, CBC
                "plan_window_size": 50,  # unit: tick
                "apply_buffer_size": 50,  # unit: tick
                "performance": {
                    "core_safety_remaining_ratio": 0,
                    "mem_safety_remaining_ratio": 0
                },
                "objective": {
                    "successful_allocation_decay": 1,
                    "allocation_multiple_core_num": False
                },
                "log": {
                    "dump_all_solution": False,
                    "dump_infeasible_solution": True,
                    "stdout_solver_message": False
                }
            },
            "start_tick": training_config["env"]["basic"]["start_tick"],
            "durations": training_config["env"]["basic"]["durations"]
        }
    },
    "optimization": {
        "actor": OptimOption(
            optim_cls="adam",
            optim_params={"lr": 0.0001}
        ),
        "critic": OptimOption(
            optim_cls="sgd",
            optim_params={"lr": 0.001}
        )
    },
    "experience_manager": {
        "capacity": 10000,
        "overwrite_type": "rolling",
        "batch_size": -1,
        "replace": False
    },
    "algorithm_config": {
        "reward_discount": 0.9,
        "train_epochs": 100,
        "gradient_iters": 1,
        "critic_loss_cls": "mse",
        "actor_loss_coefficient": 0.1
    }
}
