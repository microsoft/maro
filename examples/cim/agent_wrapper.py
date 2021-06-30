# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

from maro.rl import AgentWrapper, EpsilonGreedyExploration, MultiPhaseLinearExplorationScheduler

cim_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, cim_path)
from env_wrapper import AGENT_IDS, env_config
from meta import create_rollout_policy_func


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
        {name: func() for name, func in create_rollout_policy_func.items()},
        {name: name for name in AGENT_IDS},
        exploration_dict={f"EpsilonGreedy": epsilon_greedy},
        agent2exploration={name: "EpsilonGreedy" for name in AGENT_IDS}
    )
