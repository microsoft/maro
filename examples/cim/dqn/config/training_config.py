# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

training_config = {
    "env": {
        "scenario": "cim",
        "topology": "toy.4p_ssdd_l0.0",
        "durations": 1120,
    },
    "max_episode": 100,
    "exploration": {
        "parameter_names": ["epsilon"],
        "split": 0.5,
        "start": 0.4,
        "mid": 0.32,
        "end": 0.0
    },
    "training": {
        "min_experiences_to_train": 1024,
        "train_iter": 10,
        "batch_size": 128,
        "prioritized_sampling_by_loss": True
    },
    "group": "dqn",
    "learner_update_trigger": 2,
    "num_actors": 2,
}
