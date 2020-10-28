# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from maro.rl import ActorProxy, SimpleLearner, AgentManagerMode, TwoPhaseLinearExplorer
from maro.simulator import Env
from maro.utils import Logger

from components.agent_manager import create_dqn_agents, DQNAgentManager


def launch(config):
    def set_input_dim():
        # obtain model input dimension from state shaping configurations
        look_back = config["state_shaping"]["look_back"]
        max_ports_downstream = config["state_shaping"]["max_ports_downstream"]
        num_port_attributes = len(config["state_shaping"]["port_attributes"])
        num_vessel_attributes = len(config["state_shaping"]["vessel_attributes"])

        input_dim = (look_back + 1) * (max_ports_downstream + 1) * num_port_attributes + num_vessel_attributes
        config["agents"]["algorithm"]["input_dim"] = input_dim

    set_input_dim()

    env = Env(config.env.scenario, config.env.topology, durations=config.env.durations)
    agent_id_list = [str(agent_id) for agent_id in env.agent_idx_list]
    agent_manager = DQNAgentManager(
        name="distributed_cim_learner",
        mode=AgentManagerMode.TRAIN,
        agent_dict=create_dqn_agents(agent_id_list, config.agents),
    )

    proxy_params = {
        "group_name": os.environ["GROUP"],
        "expected_peers": {"actor": int(os.environ["NUM_ACTORS"])},
        "redis_address": ("localhost", 6379)
    }

    learner = SimpleLearner(
        trainable_agents=agent_manager,
        actor=ActorProxy(proxy_params=proxy_params),
        explorer=TwoPhaseLinearExplorer(**config.exploration),
        logger=Logger("distributed_cim_learner", auto_timestamp=False)
    )
    learner.train(config.general.max_episode)
    learner.test()
    learner.dump_models(os.path.join(os.getcwd(), "models"))


if __name__ == "__main__":
    from components.config import config
    launch(config)

