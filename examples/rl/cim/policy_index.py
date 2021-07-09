# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

from maro.rl.exploration import EpsilonGreedyExploration, MultiPhaseLinearExplorationScheduler

cim_path = os.path.dirname(os.path.realpath(__file__))
if cim_path not in sys.path:
    sys.path.insert(0, cim_path)
from dqn import get_dqn_policy
from env_wrapper import AGENT_IDS, env_config

update_trigger = {name: 128 for name in AGENT_IDS}
warmup = {name: 1 for name in AGENT_IDS}

# use agent IDs as policy names since each agent uses a separate policy
policy_func_index = {name: get_dqn_policy for name in AGENT_IDS}
agent2policy = {name: name for name in AGENT_IDS}

# exploration creators and mappings
exploration_config = {
    "last_ep": 10,
    "initial_value": 0.4,
    "final_value": 0.0,
    "splits": [(5, 0.32)]
}

def get_exploration():
    epsilon_greedy = EpsilonGreedyExploration(num_actions=env_config["wrapper"]["num_actions"])
    epsilon_greedy.register_schedule(
        scheduler_cls=MultiPhaseLinearExplorationScheduler,
        param_name="epsilon",
        **exploration_config
    )
    return epsilon_greedy


exploration_func_index = {f"EpsilonGreedy": get_exploration}
agent2exploration = {name: "EpsilonGreedy" for name in AGENT_IDS}
