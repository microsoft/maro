# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os

import numpy as np

from components.action_shaper import CIMActionShaper
from components.agent_manager import DQNAgentManager
from components.config import config
from components.experience_shaper import TruncatedExperienceShaper
from components.state_shaper import CIMStateShaper
from maro.rl import AgentMode, KStepExperienceShaper, SimpleActor, SimpleLearner, TwoPhaseLinearExplorer
from maro.simulator import Env
from maro.utils import Logger

if __name__ == "__main__":
    # Step 1: initialize a CIM environment for using a toy dataset.
    env = Env(config.env.scenario, config.env.topology, durations=config.env.durations)
    agent_id_list = [str(agent_id) for agent_id in env.agent_idx_list]

    # Step 2: create state, action and experience shapers. We also need to create an explorer here due to the
    # greedy nature of the DQN algorithm.
    state_shaper = CIMStateShaper(**config.state_shaping)
    action_shaper = CIMActionShaper(action_space=list(np.linspace(-1.0, 1.0, config.agents.algorithm.num_actions)))
    if config.experience_shaping.type == "truncated":
        experience_shaper = TruncatedExperienceShaper(**config.experience_shaping.truncated)
    else:
        experience_shaper = KStepExperienceShaper(
            reward_func=lambda mt: 1 - mt["container_shortage"] / mt["order_requirements"],
            **config.experience_shaping.k_step
        )

    exploration_config = {"epsilon_range_dict": {"_all_": config.exploration.epsilon_range},
                          "split_point_dict": {"_all_": config.exploration.split_point},
                          "with_cache": config.exploration.with_cache
                          }
    explorer = TwoPhaseLinearExplorer(agent_id_list, config.general.total_training_episodes, **exploration_config)

    # Step 3: create an agent manager.
    agent_manager = DQNAgentManager(name="cim_learner",
                                    mode=AgentMode.TRAIN_INFERENCE,
                                    agent_id_list=agent_id_list,
                                    state_shaper=state_shaper,
                                    action_shaper=action_shaper,
                                    experience_shaper=experience_shaper,
                                    explorer=explorer)

    # Step 4: Create an actor and a learner to start the training process.
    actor = SimpleActor(env=env, inference_agents=agent_manager)
    learner = SimpleLearner(trainable_agents=agent_manager, actor=actor,
                            logger=Logger("single_host_cim_learner", auto_timestamp=False))

    learner.train(total_episodes=config.general.total_training_episodes)
    learner.test()
    learner.dump_models(os.path.join(os.getcwd(), "models"))
