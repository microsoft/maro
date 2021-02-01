# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from statistics import mean

import numpy as np

from maro.simulator import Env
from maro.rl import Scheduler, SimpleLearner
from maro.utils import LogFormat, Logger, convert_dottable

from components import (
    CIMActionShaper, CIMStateShaper, SchedulerWithStopping, POAgentManager, TruncatedExperienceShaper, create_po_agents
)


def launch(config):
    # First determine the input dimension and add it to the config.
    config = convert_dottable(config)

    # Step 1: initialize a CIM environment for using a toy dataset.
    env = Env(config.env.scenario, config.env.topology, durations=config.env.durations)
    agent_id_list = [str(agent_id) for agent_id in env.agent_idx_list]

    # Step 2: create state, action and experience shapers. We also need to create an explorer here due to the
    # greedy nature of the DQN algorithm.
    state_shaper = CIMStateShaper(**config.env.state_shaping)
    action_shaper = CIMActionShaper(action_space=list(np.linspace(-1.0, 1.0, config.agents.num_actions)))
    experience_shaper = TruncatedExperienceShaper(**config.env.experience_shaping)

    # Step 3: create an agent manager.
    config["agents"]["input_dim"] = state_shaper.dim
    agent_manager = POAgentManager(
        create_po_agents(agent_id_list, config.agents),
        state_shaper=state_shaper,
        action_shaper=action_shaper,
        experience_shaper=experience_shaper,
    )

    # Step 4: Create an actor and a learner to start the training process.
    scheduler = SchedulerWithStopping(config.main_loop.max_episode, **config.main_loop.early_stopping)
    learner = SimpleLearner(env, agent_manager, scheduler)
    learner.learn()
    learner.test()
    learner.dump_models(os.path.join(os.getcwd(), "models"))


if __name__ == "__main__":
    from components.config import config
    launch(config)
