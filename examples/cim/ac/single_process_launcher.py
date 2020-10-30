# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import numpy as np

from maro.simulator import Env
from maro.rl import SimpleLearner, SimpleActor, AgentManagerMode
from maro.utils import Logger, convert_dottable

from components.action_shaper import CIMActionShaper
from components.agent_manager import create_ac_agents, ACAgentManager
from components.config import set_input_dim
from components.experience_shaper import TruncatedExperienceShaper
from components.state_shaper import CIMStateShaper


def launch(config):
    # First determine the input dimension and add it to the config.
    set_input_dim(config)
    config = convert_dottable(config)

    # Step 1: initialize a CIM environment for using a toy dataset.
    env = Env(config.env.scenario, config.env.topology, durations=config.env.durations)
    agent_id_list = [str(agent_id) for agent_id in env.agent_idx_list]

    # Step 2: create state, action and experience shapers. We also need to create an explorer here due to the
    # greedy nature of the DQN algorithm.
    state_shaper = CIMStateShaper(**config.state_shaping)
    action_shaper = CIMActionShaper(action_space=list(np.linspace(-1.0, 1.0, config.agents.algorithm.num_actions)))
    experience_shaper = TruncatedExperienceShaper(**config.experience_shaping)

    # Step 3: create an agent manager.
    agent_manager = ACAgentManager(
        name="cim_learner",
        mode=AgentManagerMode.TRAIN_INFERENCE,
        agent_dict=create_ac_agents(agent_id_list, config.agents),
        state_shaper=state_shaper,
        action_shaper=action_shaper,
        experience_shaper=experience_shaper,
    )

    # Step 4: Create an actor and a learner to start the training process.
    actor = SimpleActor(env=env, inference_agents=agent_manager)
    learner = SimpleLearner(
        trainable_agents=agent_manager, actor=actor,
        logger=Logger("single_host_cim_learner", auto_timestamp=False)
    )
    learner.train(max_episode=config.general.max_episode)
    learner.test()
    learner.dump_models(os.path.join(os.getcwd(), "models"))


if __name__ == "__main__":
    from components.config import config
    launch(config)
