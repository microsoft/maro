# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import numpy as np

from maro.rl import MultiAgentWrapper, TwoPhaseLinearParameterScheduler
from maro.simulator import Env
from maro.utils import LogFormat, Logger, convert_dottable

from examples.cim.dqn.components import (
    Actor, CIMActionShaper, CIMStateShaper, Learner, TruncatedExperienceShaper, create_dqn_agents
)


def launch(config):
    config = convert_dottable(config)
    # Step 1: Initialize a CIM environment for using a toy dataset.
    env = Env(config.env.scenario, config.env.topology, durations=config.env.durations)
    agent_id_list = [str(agent_id) for agent_id in env.agent_idx_list]
    action_space = list(np.linspace(-1.0, 1.0, config.agent.model.output_dim))

    # Step 2: Create state, action and experience shapers. We also need to create an explorer here due to the
    # greedy nature of the DQN algorithm.
    state_shaper = CIMStateShaper(**config.env.state_shaping)
    action_shaper = CIMActionShaper(action_space)
    experience_shaper = TruncatedExperienceShaper(**config.env.experience_shaping)

    # Step 3: Create agents and an agent manager.
    config.agent.model.input_dim = state_shaper.dim
    agent = MultiAgentWrapper(create_dqn_agents(agent_id_list, config.agent))

    # Step 4: Create an actor and a learner to start the training process.
    scheduler = TwoPhaseLinearParameterScheduler(config.main_loop.max_episode, **config.main_loop.exploration)
    actor = Actor(env, agent, state_shaper, action_shaper, experience_shaper)
    learner = Learner(actor, scheduler)
    learner.learn()
    agent.dump_model(os.path.join(os.getcwd(), "models"))


if __name__ == "__main__":
    from examples.cim.dqn.components.config import config
    launch(config)
