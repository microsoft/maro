# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from maro.rl import ActorProxy, AgentManagerMode, merge_experiences_with_trajectory_boundaries, SimpleLearner
from maro.simulator import Env
from maro.utils import Logger

from components.agent_manager import create_pg_agents, PGAgentManager
from components.config import config


if __name__ == "__main__":
    env = Env(config.env.scenario, config.env.topology, durations=config.env.durations)
    agent_id_list = [str(agent_id) for agent_id in env.agent_idx_list]
    agent_manager = PGAgentManager(
        name="cim_remote_learner",
        mode=AgentManagerMode.TRAIN,
        agent_dict=create_pg_agents(agent_id_list, config.agents),
    )

    proxy_params = {
        "group_name": os.environ["GROUP"],
        "expected_peers": {"actor": int(os.environ["NUM_ACTORS"])},
        "redis_address": ("localhost", 6379)
    }

    learner = SimpleLearner(
        trainable_agents=agent_manager,
        actor=ActorProxy(
            proxy_params=proxy_params,
            experience_collecting_func=merge_experiences_with_trajectory_boundaries
        ),
        logger=Logger("distributed_cim_learner", auto_timestamp=False)
    )
    learner.train(max_episode=config.general.total_training_episodes)
    learner.test()
    learner.dump_models(os.path.join(os.getcwd(), "models"))
