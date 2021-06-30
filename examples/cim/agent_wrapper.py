# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

from maro.rl import AgentWrapper, EpsilonGreedyExploration, MultiPhaseLinearExplorationScheduler

cim_path = os.path.dirname(__file__)
sys.path.insert(0, cim_path)
from env_wrapper import env_config
from meta import CIM_AGENT_IDS, CIM_CREATE_ROLLOUT_POLICY_FUNC


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
        {name: func() for name, func in CIM_CREATE_ROLLOUT_POLICY_FUNC.items()},
        {name: name for name in CIM_AGENT_IDS},
        exploration_dict={f"EpsilonGreedy": epsilon_greedy},
        agent2exploration={name: "EpsilonGreedy" for name in CIM_AGENT_IDS}
    )
