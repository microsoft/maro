# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

training_config = {
    "env": {
        "scenario": "cim",
        "topology": "global_trade.22p_l0.8",
        "durations": 1120,
    },
    "max_episode": 500,
    "train_iter": 1,
    "batch_size": 16,
    "train_freq": 1,
    "model_save_freq": 1,
    "exploration": {
        "parameter_names": ["epsilon"],
        "split": 0.5,
        "start": 0.4,
        "mid": 0.32,
        "end": 0.0
    },
    "group": "cim-gnn",
    "learner": {
        "update_trigger": 2,
        "inference_trigger": 2
    },
    "actor": {
        "num": 2,
        "receive_action_timeout": 100,  # in milliseconds
        "max_receive_action_attempts": 1,
        "max_null_actions_per_rollout": 15
    }
}
