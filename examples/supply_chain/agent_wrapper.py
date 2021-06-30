# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

from maro.rl import AgentWrapper, EpsilonGreedyExploration, LinearExplorationScheduler

sc_path = os.path.dirname(os.path.realpath(__file__))
if sc_path not in sys.path:
    sys.path.insert(0, sc_path)
from env_wrapper import NUM_ACTIONS, AGENT_IDS
from meta import create_rollout_policy_func

exploration_config = {
    "last_ep": 800,
    "initial_value": 0.8,   # Here (start: 0.4, end: 0.0) means: the exploration rate will start at 0.4 and decrease linearly to 0.0 in the last episode.
    "final_value": 0.0
}

def get_agent_wrapper():
    epsilon_greedy = EpsilonGreedyExploration(num_actions=NUM_ACTIONS)
    epsilon_greedy.register_schedule(
        scheduler_cls=LinearExplorationScheduler,
        param_name="epsilon",
        **exploration_config
    )
    return AgentWrapper(
        {name: func() for name, func in create_rollout_policy_func.items()},
        {agent_id: agent_id.split(".")[0] for agent_id in AGENT_IDS},
        exploration_dict={"consumerstore": epsilon_greedy},
        agent2exploration = {
            agent_id: "consumerstore" for agent_id in AGENT_IDS if agent_id.startswith("consumerstore")
        }
    )
