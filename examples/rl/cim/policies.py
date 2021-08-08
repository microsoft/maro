# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import importlib
import os
import sys

from maro.rl.policy import CorePolicy, PolicyUpdateOptions

cim_path = os.path.dirname(os.path.realpath(__file__))
algo_path = os.path.join(cim_path, "algorithms")

if cim_path not in sys.path:
    sys.path.insert(0, cim_path)
if algo_path not in sys.path:
    sys.path.insert(0, algo_path)

from env_wrapper import AGENT_IDS

update_options = {
    name: PolicyUpdateOptions(update_trigger=64, warmup=1024, num_epochs=10, reset_memory=False, data_parallel=False)
    for name in AGENT_IDS
}

experience_memory_config = {
    "memory_capacity": 100000,
    "random_overwrite": False
}

sampling_config = {
    "rollout": {"batch_size": 1280, "replace": True},
    "learning": {"batch_size": 128, "replace": True}
}


algorithm_type = "dqn"  # "dqn" or "ac"
module = importlib.import_module(algorithm_type)

def get_policy(rollout_only: bool = False):
    return CorePolicy(
        getattr(module, "get_algorithm")(),
        **experience_memory_config,
        sampler_kwargs=sampling_config["rollout" if rollout_only else "learning"]
    )

# use agent IDs as policy names since each agent uses a separate policy
rl_policy_func_index = {name: get_policy for name in AGENT_IDS}
agent2policy = {name: name for name in AGENT_IDS}
