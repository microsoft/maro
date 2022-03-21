# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from maro.rl.workflows.scenario import Scenario

# config variables
SCENARIO_PATH = os.path.join("rl", "supply_chain")

if __name__ == "__main__":
    scenario = Scenario(SCENARIO_PATH)
    agent2policy = scenario.agent2policy
    policy_creator = scenario.policy_creator
    policy_dict = {name: get_policy_func(name) for name, get_policy_func in policy_creator.items()}
    policy_creator = {name: lambda name: policy_dict[name] for name in policy_dict}
    env_sampler = scenario.get_env_sampler(policy_creator)
    result = env_sampler.eval()
    print(result)
    if scenario.post_evaluate:
        scenario.post_evaluate(result["info"], 0)
