# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import numpy as np

from maro.communication import Proxy
from maro.rl import BaseDistActor, MultiAgentWrapper
from maro.simulator import Env
from maro.utils import convert_dottable

from examples.cim.dqn.components import (
    Actor, CIMActionShaper, CIMStateShaper, TruncatedExperienceShaper, create_dqn_agents
)
from examples.cim.dqn.distributed.actor_client import SimpleActorClient


def launch(config, distributed_config):
    config = convert_dottable(config)
    distributed_config = convert_dottable(distributed_config)
    
    env = Env(config.env.scenario, config.env.topology, durations=config.env.durations)
    agent_id_list = [str(agent_id) for agent_id in env.agent_idx_list]
    state_shaper = CIMStateShaper(**config.env.state_shaping)
    action_shaper = CIMActionShaper(action_space=list(np.linspace(-1.0, 1.0, config.agent.model.output_dim)))
    experience_shaper = TruncatedExperienceShaper(**config.env.experience_shaping)
    
    inference_mode = distributed_config.inference_mode
    redis_address = distributed_config.redis.hostname, distributed_config.redis.port
    if inference_mode == "remote":
        agent_proxy = Proxy(
            group_name=distributed_config.group,
            component_type="actor_client",
            expected_peers=distributed_config.peers.actor,
            redis_address=redis_address,
            max_retries=20,
            driver_parameters={"receive_timeout": distributed_config.receive_action_timeout}
        )
        actor = SimpleActorClient(
            env, agent_proxy, state_shaper, action_shaper, experience_shaper,
            receive_action_timeout=distributed_config.receive_action_timeout,
            max_receive_action_attempts=distributed_config.max_receive_action_attempts
        )
    elif inference_mode == "local":
        config.agent.model.input_dim = state_shaper.dim
        agent = MultiAgentWrapper(create_dqn_agents(agent_id_list, config.agent))
        actor = Actor(env, agent, state_shaper, action_shaper, experience_shaper)
    else:
        raise ValueError(f'Supported distributed training modes: "local", "remote", got {inference_mode}')

    proxy = Proxy(
        group_name=distributed_config.group,
        component_type="actor",
        expected_peers=distributed_config.peers.actor,
        redis_address=redis_address,
        max_retries=20
    )
    
    dist_actor = BaseDistActor(actor, proxy)
    dist_actor.run()


if __name__ == "__main__":
    from examples.cim.dqn.components.config import config, distributed_config
    launch(config=config, distributed_config=distributed_config)
