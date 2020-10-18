# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import numpy as np

from maro.simulator import Env
from maro.rl import SimpleLearner, SimpleActor, AgentMode
from maro.utils import Logger

from components.action_shaper import CIMActionShaper
from components.agent_manager import PGAgentManager
from components.config import config
from components.experience_shaper import TruncatedExperienceShaper
from components.state_shaper import CIMStateShaper


if __name__ == "__main__":
    # Step 1: initialize a CIM environment for using a toy dataset.
    env = Env(config.env.scenario, config.env.topology, durations=config.env.durations)
    agent_id_list = [str(agent_id) for agent_id in env.agent_idx_list]

    # Step 2: create state, action and experience shapers. We also need to create an explorer here due to the
    # greedy nature of the DQN algorithm.
    state_shaper = CIMStateShaper(**config.state_shaping)
    action_shaper = CIMActionShaper(action_space=list(np.linspace(-1.0, 1.0, config.agents.algorithm.num_actions)))
    experience_shaper = TruncatedExperienceShaper(**config.experience_shaping.truncated)

    # Step 3: create an agent manager.
    agent_manager = PGAgentManager(
        name="cim_learner",
        mode=AgentMode.TRAIN_INFERENCE,
        agent_id_list=agent_id_list,
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
    learner.train(
        total_episodes=config.general.total_training_episodes,
        model_dump_dir=os.path.join(os.getcwd(), "models")
    )
    learner.test()
