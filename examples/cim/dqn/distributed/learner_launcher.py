# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import time

from maro.communication import Proxy
from maro.rl import MultiAgentWrapper, TwoPhaseLinearParameterScheduler
from maro.simulator import Env
from maro.utils import convert_dottable

from examples.cim.dqn.components import CIMStateShaper, create_dqn_agents
from examples.cim.dqn.distributed.learner import SimpleDistLearner


def launch(config, distributed_config):
    config = convert_dottable(config)
    distributed_config = convert_dottable(distributed_config)
    env = Env(config.env.scenario, config.env.topology, durations=config.env.durations)
    config.agent.model.input_dim = CIMStateShaper(**config.env.state_shaping).dim
    config.agent.names = [str(agent_id) for agent_id in env.agent_idx_list]
    agent = MultiAgentWrapper(create_dqn_agents(config.agent))
    scheduler = TwoPhaseLinearParameterScheduler(config.main_loop.max_episode, **config.main_loop.exploration)

    inference = distributed_config.inference_mode == "remote"
    if inference:
        distributed_config.peers.learner.actor_client = distributed_config.peers.learner.actor
    proxy = Proxy(
        group_name=distributed_config.group,
        component_type="learner",
        expected_peers=distributed_config.peers.learner,
        redis_address=(distributed_config.redis.hostname, distributed_config.redis.port),
        max_retries=15
    )
    
    learner = SimpleDistLearner(
        agent, scheduler, proxy,
        update_trigger=distributed_config.update_trigger,
        inference=inference,
        inference_trigger=distributed_config.inference_trigger,
    )

    time.sleep(5)
    learner.learn()
    learner.dump_model(os.path.join(os.getcwd(), "models"))
    learner.exit()


if __name__ == "__main__":
    from examples.cim.dqn.components.config import config, distributed_config
    launch(config=config, distributed_config=distributed_config)
