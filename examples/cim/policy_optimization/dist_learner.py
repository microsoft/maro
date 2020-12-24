# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from maro.rl import ActorProxy, AgentManagerMode, Scheduler, SimpleLearner, merge_experiences_with_trajectory_boundaries
from maro.simulator import Env
from maro.utils import Logger, convert_dottable

from components import CIMStateShaper, POAgentManager, create_po_agents


def launch(config):
    config = convert_dottable(config)
    env = Env(config.env.scenario, config.env.topology, durations=config.env.durations)
    agent_id_list = [str(agent_id) for agent_id in env.agent_idx_list]
    config["agents"]["input_dim"] = CIMStateShaper(**config.env.state_shaping).dim
    agent_manager = POAgentManager(
        name="cim_learner",
        mode=AgentManagerMode.TRAIN,
        agent_dict=create_po_agents(agent_id_list, config.agents)
    )

    proxy_params = {
        "group_name": os.environ["GROUP"],
        "expected_peers": {"actor": int(os.environ["NUM_ACTORS"])},
        "redis_address": ("localhost", 6379)
    }

    learner = SimpleLearner(
        agent_manager=agent_manager,
        actor=ActorProxy(
            proxy_params=proxy_params, experience_collecting_func=merge_experiences_with_trajectory_boundaries
        ),
        scheduler=Scheduler(config.main_loop.max_episode),
        logger=Logger("cim_learner", auto_timestamp=False)
    )
    learner.learn()
    learner.test()
    learner.dump_models(os.path.join(os.getcwd(), "models"))
    learner.exit()


if __name__ == "__main__":
    from components.config import config
    launch(config)
