# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from components.agent_manager import DQNAgentManager
from components.config import config
from components.state_shaper import CIMStateShaper
from maro.rl import ActorProxy, AgentMode, SimpleLearner, TwoPhaseLinearExplorer
from maro.simulator import Env
from maro.utils import Logger

if __name__ == "__main__":
    env = Env(config.env.scenario, config.env.topology, durations=config.env.durations)
    agent_id_list = [str(agent_id) for agent_id in env.agent_idx_list]
    state_shaper = CIMStateShaper(**config.state_shaping)
    exploration_config = {"epsilon_range_dict": {"_all_": config.exploration.epsilon_range},
                          "split_point_dict": {"_all_": config.exploration.split_point},
                          "with_cache": config.exploration.with_cache
                          }
    explorer = TwoPhaseLinearExplorer(agent_id_list, config.general.total_training_episodes, **exploration_config)
    agent_manager = DQNAgentManager(name="cim_remote_learner", agent_id_list=agent_id_list, mode=AgentMode.TRAIN,
                                    state_shaper=state_shaper, explorer=explorer)

    proxy_params = {
        "group_name": config.distributed.group_name,
        "expected_peers": config.distributed.learner.peer,
        "redis_address": (config.distributed.redis.host_name, config.distributed.redis.port),
        "max_retries": 10
    }
    learner = SimpleLearner(trainable_agents=agent_manager,
                            actor=ActorProxy(proxy_params=proxy_params),
                            logger=Logger("distributed_cim_learner", auto_timestamp=False))
    learner.train(total_episodes=config.general.total_training_episodes)
    learner.test()
    learner.dump_models(os.path.join(os.getcwd(), "models"))
