# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

training_config = {
    "seed": 1024,
    "env": {
        "scenario": "cim",
        "topology": "toy.5p_ssddd_l0.0",
        "durations": 1120,
    },
    "max_episode": 500,
    "train_freq": 1,
    "model_save_freq": 1,
    "group": "cim-gnn-4",
    "learner": {
        "update_trigger": 24,
        "inference_trigger": 24
    },
    "actor": {
        "num": 24,
        "receive_action_timeout": 500,  # in milliseconds
        "max_receive_action_attempts": 1,
        "max_null_actions_per_rollout": 15
    }
}
