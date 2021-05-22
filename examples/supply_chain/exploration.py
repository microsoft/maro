# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os.path import dirname, realpath

from maro.rl import EpsilonGreedyExploration, LinearExplorationScheduler, MultiPhaseLinearExplorationScheduler

sc_code_dir = dirname(realpath(__file__))
sys.path.insert(0, sc_code_dir)
from config import config

MultiPhaseLinearExplorationScheduler

exploration = EpsilonGreedyExploration(config["policy"]["consumer"]["model"]["network"]["output_dim"])
exploration.register_schedule(
    LinearExplorationScheduler, "epsilon", config["exploration"]["last_ep"],
    initial_value=config["exploration"]["initial_value"],
    final_value=config["exploration"]["final_value"]
)

exploration_dict = {"consumerstore": exploration}

# all agents shared the same exploration object
agent2exploration = {agent_id: "consumerstore" for agent_id in config["agent_ids"] if agent_id.startswith('consumerstore')}
