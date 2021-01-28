# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import time

from maro.communication import Proxy
from maro.rl import InferenceLearner, SimpleDistLearner, TwoPhaseLinearParameterScheduler, concat_experiences_by_agent
from maro.simulator import Env
from maro.utils import convert_dottable

from examples.cim.dqn.components import CIMStateShaper, DQNAgentManager, create_dqn_agents


def launch(config, distributed_config):
    config = convert_dottable(config)
    distributed_config = convert_dottable(distributed_config)
    env = Env(config.env.scenario, config.env.topology, durations=config.env.durations)
    agent_id_list = [str(agent_id) for agent_id in env.agent_idx_list]

    config["agents"]["algorithm"]["input_dim"] = CIMStateShaper(**config.env.state_shaping).dim
    agent_manager = DQNAgentManager(
        name="distributed_cim_learner",
        mode=AgentManagerMode.TRAIN,
        agents=create_dqn_agents(agent_id_list, config.agents)
    )

    scheduler = TwoPhaseLinearParameterScheduler(config.main_loop.max_episode, **config.main_loop.exploration)

    distributed_mode = os.environ.get("MODE", distributed_config.mode)
    expected_peers={"actor": int(os.environ.get("NUM_ACTORS", distributed_config.num_actors))}
    if distributed_mode == "seed":
        expected_peers["agent_manager"] = expected_peers["actor"]
    proxy_params = {
        "group_name": os.environ["GROUP"] if "GROUP" in os.environ else distributed_config.group,
        "component_type": "learner",
        "expected_peers": expected_peers,
        "redis_address": (distributed_config.redis.hostname, distributed_config.redis.port),
        "max_retries": 15
    }
    
    if distributed_mode == "seed":
        learner = InferenceLearner(
            agent_manager, scheduler, Proxy(**proxy_params), concat_experiences_by_agent,
            choose_action_trigger=distributed_config.choose_action_trigger,
            update_trigger=distributed_config.update_trigger
        )
    elif distributed_mode == "simple":
        learner = SimpleDistLearner(
            agent_manager, scheduler, Proxy(**proxy_params), concat_experiences_by_agent,
            update_trigger=distributed_config.update_trigger
        )
    else:
        raise ValueError(f'Supported distributed training modes: "simple", "seed", got {distributed_mode}')

    time.sleep(5)
    learner.learn()
    learner.test()
    learner.dump_models(os.path.join(os.getcwd(), "models"))
    learner.exit()


if __name__ == "__main__":
    from examples.cim.dqn.components.config import config, distributed_config
    launch(config=config, distributed_config=distributed_config)
