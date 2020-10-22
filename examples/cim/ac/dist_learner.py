# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from maro.rl import ActorProxy, SimpleLearner, AgentMode, AgentManagerMode
from maro.simulator import Env
from maro.utils import Logger

from components.agent_manager import create_ac_agents, ACAgentManager
from components.config import config


if __name__ == "__main__":
    env = Env(config.env.scenario, config.env.topology, durations=config.env.durations)
    agent_id_list = [str(agent_id) for agent_id in env.agent_idx_list]
    agent_manager = ACAgentManager(
        name="cim_remote_learner",
        mode=AgentManagerMode.TRAIN,
        agent_dict=create_ac_agents(agent_id_list, AgentMode.TRAIN)
    )

    proxy_params = {
        "group_name": config.distributed.group_name,
        "expected_peers": config.distributed.learner.peer,
        "redis_address": (config.distributed.redis.host_name, config.distributed.redis.port)
    }
    learner = SimpleLearner(
        trainable_agents=agent_manager,
        actor=ActorProxy(proxy_params=proxy_params),
        logger=Logger("distributed_cim_learner", auto_timestamp=False)
    )
    learner.train(total_episodes=config.general.total_training_episodes)
    learner.test()
    learner.dump_models(os.path.join(os.getcwd(), "models"))
