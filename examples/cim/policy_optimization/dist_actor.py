# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import numpy as np

from maro.simulator import Env
from maro.rl import AgentManagerMode, SimpleActor, ActorWorker
from maro.utils import convert_dottable

from components import CIMActionShaper, CIMStateShaper, POAgentManager, TruncatedExperienceShaper, create_po_agents


def launch(config):
    config = convert_dottable(config)
    env = Env(config.env.scenario, config.env.topology, durations=config.env.durations)
    agent_id_list = [str(agent_id) for agent_id in env.agent_idx_list]
    state_shaper = CIMStateShaper(**config.env.state_shaping)
    action_shaper = CIMActionShaper(action_space=list(np.linspace(-1.0, 1.0, config.agents.num_actions)))
    experience_shaper = TruncatedExperienceShaper(**config.env.experience_shaping)

    config["agents"]["input_dim"] = state_shaper.dim
    agent_manager = POAgentManager(
        name="cim_actor",
        mode=AgentManagerMode.INFERENCE,
        agent_dict=create_po_agents(agent_id_list, config.agents),
        state_shaper=state_shaper,
        action_shaper=action_shaper,
        experience_shaper=experience_shaper,
    )
    proxy_params = {
        "group_name": os.environ["GROUP"],
        "expected_peers": {"learner": 1},
        "redis_address": ("localhost", 6379)
    }
    actor_worker = ActorWorker(
        local_actor=SimpleActor(env=env, agent_manager=agent_manager),
        proxy_params=proxy_params
    )
    actor_worker.launch()


if __name__ == "__main__":
    from components.config import config
    launch(config)
