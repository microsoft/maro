# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import numpy as np


from maro.rl import ActorWorker, AgentManagerMode, SimpleActor
from maro.simulator import Env
from maro.utils import convert_dottable

from components import CIMActionShaper, CIMStateShaper, DQNAgentManager, TruncatedExperienceShaper, create_dqn_agents


def launch(config, distributed_config):
    config = convert_dottable(config)
    distributed_config = convert_dottable(distributed_config)
    env = Env(config.env.scenario, config.env.topology, durations=config.env.durations)
    agent_id_list = [str(agent_id) for agent_id in env.agent_idx_list]
    state_shaper = CIMStateShaper(**config.env.state_shaping)
    action_shaper = CIMActionShaper(action_space=list(np.linspace(-1.0, 1.0, config.agents.algorithm.num_actions)))
    experience_shaper = TruncatedExperienceShaper(**config.env.experience_shaping)

    config["agents"]["algorithm"]["input_dim"] = state_shaper.dim
    agent_manager = DQNAgentManager(
        name="cim_actor",
        mode=AgentManagerMode.INFERENCE,
        agent_dict=create_dqn_agents(agent_id_list, config.agents),
        state_shaper=state_shaper,
        action_shaper=action_shaper,
        experience_shaper=experience_shaper
    )
    proxy_params = {
        "group_name": os.environ["GROUP"] if "GROUP" in os.environ else distributed_config.group,
        "expected_peers": {"learner": 1},
        "redis_address": (distributed_config.redis.hostname, distributed_config.redis.port),
        "max_retries": 15
    }
    actor_worker = ActorWorker(
        local_actor=SimpleActor(env=env, agent_manager=agent_manager),
        proxy_params=proxy_params
    )
    actor_worker.launch()


if __name__ == "__main__":
    from components.config import config, distributed_config
    launch(config=config, distributed_config=distributed_config)
