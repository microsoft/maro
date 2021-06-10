# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

training_config = {
    "experiment_name": "vm_schedueling_dqn",
    "num_episodes": 100,
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
            "alpha": 0.0,
            "beta": 1.0,
            "pm_num": 8,
            "durations": 200,
            "vm_state_path": "../data/train_vm_states.npy",
            "vm_window_size": 1,
            "pm_window_size": 1,
            "window_type": "fix",
            "window_size": 1,
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
            "alpha": 0.0,
            "beta": 1.0,
            "pm_num": 8,
            "durations": 300,
            "vm_state_path": "../data/test_vm_states.npy",
            "vm_window_size": 1,
            "pm_window_size": 1,
            "window_type": "fix",
            "window_size": 1,
            "gamma": 0.9
        }
    },
    "seed": 666,
    "exploration": {
        "last_ep": 400,
        "initial_value": 0.4,
        "final_value": 0.0,
        "splits": [[100, 0.32]]
    },
    "training": {
        "min_experiences_to_train": 0,
        "train_iter": 100,
        "batch_size": 256,
        "prioritized_sampling_by_loss": True,
        "validate_interval": 10
    }
}
