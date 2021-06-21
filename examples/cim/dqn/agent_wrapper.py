# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

from maro.rl import AgentWrapper, EpsilonGreedyExploration, MultiPhaseLinearExplorationScheduler

dqn_path = os.path.dirname(os.path.realpath(__file__))  # DQN directory
cim_path = os.path.dirname(dqn_path)  # CIM example directory
sys.path.insert(0, cim_path)
sys.path.insert(0, dqn_path)
from general import AGENT_IDS, NUM_ACTIONS, config, log_dir
from policy import get_independent_policy_for_rollout

def get_agent_wrapper():
    epsilon_greedy = EpsilonGreedyExploration(num_actions=NUM_ACTIONS)
    epsilon_greedy.register_schedule(
        scheduler_cls=MultiPhaseLinearExplorationScheduler,
        param_name="epsilon",
        last_ep=config["num_episodes"],
        **config["exploration"]
    )
    return AgentWrapper(
        policies=[get_independent_policy_for_rollout(i) for i in AGENT_IDS],
        agent2policy={i: i for i in AGENT_IDS},
        exploration_dict={f"EpsilonGreedy": epsilon_greedy},
        agent2exploration={i: "EpsilonGreedy" for i in AGENT_IDS},
        log_dir=log_dir
    )
