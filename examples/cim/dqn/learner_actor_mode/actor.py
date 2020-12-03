# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import numpy as np

from maro.rl import AgentManagerMode, LearnerActorComponent, KStepExperienceShaper
from maro.simulator import Env
from maro.utils import convert_dottable

from examples.cim.dqn.components.action_shaper import CIMActionShaper
from examples.cim.dqn.components.agent_manager import DQNAgentManager, create_dqn_agents
from examples.cim.dqn.components.config import set_input_dim
from examples.cim.dqn.components.experience_shaper import TruncatedExperienceShaper
from examples.cim.dqn.components.state_shaper import CIMStateShaper


def launch(config, distributed_config):
    set_input_dim(config)
    config = convert_dottable(config)
    distributed_config = convert_dottable(distributed_config)
    env = Env(config.env.scenario, config.env.topology, durations=config.env.durations)
    agent_id_list = [str(agent_id) for agent_id in env.agent_idx_list]
    state_shaper = CIMStateShaper(**config.state_shaping)
    action_shaper = CIMActionShaper(action_space=list(np.linspace(-1.0, 1.0, config.agents.algorithm.num_actions)))
    if config.experience_shaping.type == "truncated":
        experience_shaper = TruncatedExperienceShaper(**config.experience_shaping.truncated)
    else:
        experience_shaper = KStepExperienceShaper(
            reward_func=lambda mt: 1 - mt["container_shortage"] / mt["order_requirements"],
            **config.experience_shaping.k_step
        )

    distributed_mode = os.environ.get("MODE", distributed_config.mode)
    if distributed_mode == "seed":
        from maro.rl import SEEDActor
        actor = SEEDActor(
            env, state_shaper, action_shaper, experience_shaper,
            group_name=os.environ["GROUP"] if "GROUP" in os.environ else distributed_config.group,
            expected_peers={LearnerActorComponent.LEARNER.value: 1},
            redis_address=(distributed_config.redis.hostname, distributed_config.redis.port),
            max_retries=15
        )
    elif distributed_mode == "simple":
        from maro.rl import SimpleActor
        agent_manager = DQNAgentManager(
            name="distributed_cim_actor",
            mode=AgentManagerMode.INFERENCE,
            agent_dict=create_dqn_agents(agent_id_list, config.agents),
            state_shaper=state_shaper,
            action_shaper=action_shaper,
            experience_shaper=experience_shaper
        )
        actor = SimpleActor(
            env, agent_manager,
            group_name=os.environ.get("GROUP", distributed_config.group),
            expected_peers={LearnerActorComponent.LEARNER.value: 1},
            redis_address=(distributed_config.redis.hostname, distributed_config.redis.port),
            max_retries=15
        )
    else:
        raise ValueError(f'Supported distributed training modes: "simple", "seed", got {distributed_mode}')

    actor.launch()


if __name__ == "__main__":
    from examples.cim.dqn.components.config import config, distributed_config
    launch(config=config, distributed_config=distributed_config)
