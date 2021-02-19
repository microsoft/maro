# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

training_config = {
    "env": {
        "scenario": "cim",
        "topology": "toy.4p_ssdd_l0.0",
        "durations": 1120,
    },
    "max_episode": 100,
    "k": 5,
    "warmup_ep": 20,
    "perf_thresh": 0.95
}
