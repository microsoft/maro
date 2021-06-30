# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

from maro.rl import AgentWrapper, EpsilonGreedyExploration, LinearExplorationScheduler, NullPolicy

sc_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, sc_path)
from dqn import get_dqn_policy_for_rollout
from or_policies import (
    get_consumer_baseline_policy, get_consumer_eoq_policy, get_consumer_minmax_policy, get_producer_baseline_policy
)


def get_policy_mapping(config):
    # policy_ids = ["consumerstore", "consumer", "producer", "facility", "product", "productstore"]
    policies = [
        get_consumer_min_policy("consumer", config["policy"]["consumer"]),
        get_base_producer_policy("producer", config["policy"]["producer"]),
        get_dqn_policy("consumerstore", config["policy"]["consumerstore"]),
        NullPolicy(name="facility"),
        NullPolicy(name="product"),
        NullPolicy(name="productstore")
    ]

    agent2policy = {agent_id: agent_id.split(".")[0] for agent_id in config["agent_id_list"]}
    return policies, agent2policy

def get_replay_agent_ids(agent_id_list) -> List[str]:
    replay_agent_ids = [agent_id for agent_id in agent_id_list if agent_id.startswith("consumerstore")]
    return replay_agent_ids


def get_exploration_mapping(config) -> (dict, dict):
    exploration = EpsilonGreedyExploration(
        num_actions=config["policy"]["consumer"]["model"]["network"]["output_dim"]
    )
    exploration.register_schedule(
        scheduler_cls=LinearExplorationScheduler,
        param_name="epsilon",
        last_ep=config["exploration"]["last_ep"],
        initial_value=config["exploration"]["initial_value"],
        final_value=config["exploration"]["final_value"]
    )

    exploration_dict = {"consumerstore": exploration}
    agent2exploration = {
        agent_id: "consumerstore"
        for agent_id in config["agent_id_list"] if agent_id.startswith("consumerstore")
    }

    return exploration_dict, agent2exploration


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
        {name: get_dqn_policy_for_rollout() for name in CIM_POLICY_NAMES},
        {name: name for name in CIM_POLICY_NAMES},
        exploration_dict={f"EpsilonGreedy": epsilon_greedy},
        agent2exploration={name: "EpsilonGreedy" for name in CIM_POLICY_NAMES}
    )
