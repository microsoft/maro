# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

from maro.rl.exploration import EpsilonGreedyExploration, LinearExplorationScheduler
from maro.rl.learning import AgentWrapper

sc_path = os.path.dirname(os.path.realpath(__file__))
if sc_path not in sys.path:
    sys.path.insert(0, sc_path)
from env_wrapper import NUM_ACTIONS, AGENT_IDS
from policy_index import NUM_RL_POLICIES, rollout_policy_func_index



def get_agent_wrapper():
    
    consumerstores = [agent_id for agent_id in AGENT_IDS if agent_id.startswith("consumerstore")]
    agent2policy = {
        agent_id: agent_id.split(".")[0] for agent_id in AGENT_IDS if not agent_id.startswith("consumerstore")
    }
    for i, agent_id in enumerate(consumerstores):
        agent2policy[agent_id] = f"consumerstore-{i % NUM_RL_POLICIES}"

    return AgentWrapper(
        {name: func() for name, func in rollout_policy_func_index.items()},
        agent2policy,
        exploration_dict={"consumerstore": epsilon_greedy},
        agent2exploration = {
            agent_id: "consumerstore" for agent_id in AGENT_IDS if agent_id.startswith("consumerstore")
        }
    )
