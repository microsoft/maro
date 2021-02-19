# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

training_config = {
    "env": {
        "scenario": "cim",
        "topology": "toy.4p_ssdd_l0.0",
        "durations": 1120,
    },
    "max_episode": 200,
    "exploration": {
        "parameter_names": ["epsilon"],
        "split": 0.5,
        "start": 0.4,
        "mid": 0.32,
        "end": 0.0
    },
    "group": "cim-dqn",
    "learner_update_trigger": 2,
    "num_actors": 2
}
