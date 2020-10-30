# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import numpy as np

from maro.simulator import Env
from maro.rl import AgentManagerMode, SimpleActor, ActorWorker

from components.action_shaper import CIMActionShaper
from components.agent_manager import create_pg_agents, PGAgentManager
from components.config import config
from components.experience_shaper import TruncatedExperienceShaper
from components.state_shaper import CIMStateShaper


if __name__ == "__main__":
    env = Env(config.env.scenario, config.env.topology, durations=config.env.durations)
    agent_id_list = [str(agent_id) for agent_id in env.agent_idx_list]
    state_shaper = CIMStateShaper(**config.state_shaping)
    action_shaper = CIMActionShaper(action_space=list(np.linspace(-1.0, 1.0, config.agents.algorithm.num_actions)))
    experience_shaper = TruncatedExperienceShaper(**config.experience_shaping)
    agent_manager = PGAgentManager(
        name="cim_remote_actor",
        mode=AgentManagerMode.INFERENCE,
        agent_dict=create_pg_agents(agent_id_list, config.agents),
        state_shaper=state_shaper,
        action_shaper=action_shaper,
        experience_shaper=experience_shaper
    )
    proxy_params = {
        "group_name": os.environ["GROUP"],
        "expected_peers": {"learner": 1},
        "redis_address": ("localhost", 6379)
    }
    actor_worker = ActorWorker(
        local_actor=SimpleActor(env=env, inference_agents=agent_manager),
        proxy_params=proxy_params
    )
    actor_worker.launch()
