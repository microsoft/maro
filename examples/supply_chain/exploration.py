# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl import EpsilonGreedyExploration, LinearExplorationScheduler

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
