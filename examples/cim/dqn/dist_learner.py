# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from maro.rl import ActorProxy, SimpleLearner, TwoPhaseLinearParameterScheduler, concat_experiences_by_agent
from maro.simulator import Env
from maro.utils import Logger, convert_dottable

from components import CIMStateShaper, DQNAgentManager, create_dqn_agents


def launch(config, distributed_config):
    config = convert_dottable(config)
    distributed_config = convert_dottable(distributed_config)
    env = Env(config.env.scenario, config.env.topology, durations=config.env.durations)
    agent_id_list = [str(agent_id) for agent_id in env.agent_idx_list]

    config["agents"]["algorithm"]["input_dim"] = CIMStateShaper(**config.env.state_shaping).dim
    agent_manager = DQNAgentManager(create_dqn_agents(agent_id_list, config.agents))

    proxy_params = {
        "group_name": os.environ["GROUP"] if "GROUP" in os.environ else distributed_config.group,
        "expected_peers": {
            "actor": int(os.environ["NUM_ACTORS"] if "NUM_ACTORS" in os.environ else distributed_config.num_actors)
        },
        "redis_address": (distributed_config.redis.hostname, distributed_config.redis.port),
        "max_retries": 15
    }

    learner = SimpleLearner(
        agent_manager=agent_manager,
        actor=ActorProxy(proxy_params=proxy_params, experience_collecting_func=concat_experiences_by_agent),
        scheduler=TwoPhaseLinearParameterScheduler(config.main_loop.max_episode, **config.main_loop.exploration),
        logger=Logger("cim_learner", auto_timestamp=False)
    )
    learner.learn()
    learner.test()
    learner.dump_models(os.path.join(os.getcwd(), "models"))
    learner.exit()


if __name__ == "__main__":
    from components.config import config, distributed_config
    launch(config=config, distributed_config=distributed_config)
