# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
from os.path import dirname, join, realpath

from maro.rl.training.utils import get_latest_ep
from maro.rl.workflows.scenario import Scenario

# config variables
SCENARIO_NAME = "cim"
SCENARIO_PATH = join(dirname(dirname(realpath(__file__))), SCENARIO_NAME, "rl")
LOAD_PATH = join(dirname(SCENARIO_PATH), "checkpoints")
LOAD_EPISODE = None

if __name__ == "__main__":
    scenario = Scenario(SCENARIO_PATH)
    policy_creator = scenario.policy_creator
    policy_dict = {name: get_policy_func(name) for name, get_policy_func in policy_creator.items()}
    policy_creator = {name: lambda name: policy_dict[name] for name in policy_dict}

    env_sampler = scenario.env_sampler_creator(policy_creator)

    if LOAD_PATH is not None:
        ep = LOAD_EPISODE if LOAD_EPISODE is not None else get_latest_ep(LOAD_PATH)
        path = os.path.join(LOAD_PATH, str(ep))

        loaded = env_sampler.load_policy_state(path)
        print(f"Loaded policies {loaded} into env sampler from {path}")

    result = env_sampler.eval()
    if scenario.post_evaluate:
        scenario.post_evaluate(result["info"], 0)
