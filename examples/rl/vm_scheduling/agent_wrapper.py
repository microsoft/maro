# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

from maro.rl.exploration import EpsilonGreedyExploration, MultiPhaseLinearExplorationScheduler
from maro.rl.learning import AgentWrapper

cim_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, cim_path)
from env_wrapper import AGENT_IDS, env_config
from policy_index import policy_func_index


exploration_config = {
    "last_ep": 10,
    "initial_value": 0.4,
    "final_value": 0.0,
    "splits": [(5, 0.32)]
}

def get_agent_wrapper():
    epsilon_greedy = EpsilonGreedyExploration(num_actions=env_config["wrapper"]["num_actions"])
    epsilon_greedy.register_schedule(
        scheduler_cls=MultiPhaseLinearExplorationScheduler,
        param_name="epsilon",
        **exploration_config
    )
    return AgentWrapper(
        {name: func(learning=False) for name, func in policy_func_index.items()},
        {name: name for name in AGENT_IDS},
        exploration_dict={f"EpsilonGreedy": epsilon_greedy},
        agent2exploration={name: "EpsilonGreedy" for name in AGENT_IDS}
    )
