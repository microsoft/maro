# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import numpy as np

from maro.rl import AgentManagerMode, SimpleActor, SimpleLearner, LinearParameterScheduler
from maro.simulator import Env
from maro.utils import LogFormat, Logger, convert_dottable

from components import CIMActionShaper, CIMStateShaper, DDPGAgentManager, TruncatedExperienceShaper, create_ddpg_agents


def launch(config):
    config = convert_dottable(config)
    # Step 1: Initialize a CIM environment for using a toy dataset.
    env = Env(config.env.scenario, config.env.topology, durations=config.env.durations)
    agent_id_list = [str(agent_id) for agent_id in env.agent_idx_list]

    # Step 2: Create state, action and experience shapers. We also need to create an explorer here due to the
    # greedy nature of the DDPG algorithm.
    state_shaper = CIMStateShaper(**config.env.state_shaping)
    action_shaper = CIMActionShaper()
    experience_shaper = TruncatedExperienceShaper(**config.env.experience_shaping)

    # Step 3: Create agents and an agent manager.
    config["agents"]["input_dim"] = state_shaper.dim
    agent_manager = DDPGAgentManager(
        name="cim_learner",
        mode=AgentManagerMode.TRAIN_INFERENCE,
        agent_dict=create_ddpg_agents(agent_id_list, config.agents),
        state_shaper=state_shaper,
        action_shaper=action_shaper,
        experience_shaper=experience_shaper
    )

    # Step 4: Create an actor and a learner to start the training process.
    scheduler = LinearParameterScheduler(config.main_loop.max_episode, **config.main_loop.exploration)

    actor = SimpleActor(env, agent_manager)
    learner = SimpleLearner(
        agent_manager, actor, scheduler,
        logger=Logger("cim_learner", format_=LogFormat.simple, auto_timestamp=False)
    )
    learner.learn()
    learner.test()
    learner.dump_models(os.path.join(os.getcwd(), "models"))


if __name__ == "__main__":
    from components.config import config
    launch(config)
