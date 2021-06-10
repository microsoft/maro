# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

training_config = {
    "experiment_name": "vm_scheduling_ac",
    "num_episodes": 400,
    "num_steps": -1,
    "eval_schedule": 10,
    "auxiliary_prob": 0.1,
    "env": {
        "basic": {
            "scenario": "vm_scheduling",
            "topology": "azure.2019.10k.short",
            "start_tick": 0,
            "durations": 200,
            "snapshot_resolution": 1
        },
        "wrapper": {
            "training": True,
            "alpha": 1.0,
            "beta": 0.0,
            "pm_num": 8,
            "durations": 200,
            "vm_window_size": 1,
            "pm_window_size": 1,
            "gamma": 0.9
        }
    },
    "eval_env": {
        "basic": {
            "scenario": "vm_scheduling",
            "topology": "azure.2019.10k.short.test",
            "start_tick": 0,
            "durations": 300,
            "snapshot_resolution": 1
        },
        "wrapper": {
            "training": False,
            "alpha": 1.0,
            "beta": 0.0,
            "pm_num": 8,
            "durations": 300,
            "vm_window_size": 1,
            "pm_window_size": 1,
            "gamma": 0.9
        }
    },
    "seed": 666
}
