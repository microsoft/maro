# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import io
import yaml

import numpy as np

from maro.simulator import Env
from maro.rl import SimpleLearner, SimpleActor, AgentMode, KStepExperienceShaper, TwoPhaseLinearExplorer
from maro.utils import Logger, convert_dottable
from examples.ecr.dqn.components.state_shaper import ECRStateShaper
from examples.ecr.dqn.components.action_shaper import ECRActionShaper
from examples.ecr.dqn.components.experience_shaper import TruncatedExperienceShaper
from examples.ecr.dqn.components.agent_manager import DQNAgentManager


with io.open("config.yml", "r") as in_file:
    raw_config = yaml.safe_load(in_file)
    config = convert_dottable(raw_config)


if __name__ == "__main__":
    env = Env(config.env.scenario, config.env.topology, durations=config.env.durations)
    agent_id_list = [str(agent_id) for agent_id in env.agent_idx_list]
    state_shaper = ECRStateShaper(**config.state_shaping)
    action_shaper = ECRActionShaper(action_space=list(np.linspace(-1.0, 1.0, config.agents.algorithm.num_actions)))
    if config.experience_shaping.type == "truncated":
        experience_shaper = TruncatedExperienceShaper(**config.experience_shaping.truncated)
    else:
        experience_shaper = KStepExperienceShaper(reward_func=lambda mt: mt["perf"], **config.experience_shaping.k_step)

    exploration_config = {"epsilon_range_dict": {"_all_": config.exploration.epsilon_range},
                          "split_point_dict": {"_all_": config.exploration.split_point},
                          "with_cache": config.exploration.with_cache
                          }
    explorer = TwoPhaseLinearExplorer(agent_id_list, config.general.total_training_episodes, **exploration_config)
    agent_manager = DQNAgentManager(name="ecr_learner",
                                    mode=AgentMode.TRAIN_INFERENCE,
                                    agent_id_list=agent_id_list,
                                    state_shaper=state_shaper,
                                    action_shaper=action_shaper,
                                    experience_shaper=experience_shaper,
                                    explorer=explorer)
    learner = SimpleLearner(trainable_agents=agent_manager,
                            actor=SimpleActor(env=env, inference_agents=agent_manager),
                            logger=Logger("single_host_ecr_learner", auto_timestamp=False))

    learner.train(total_episodes=config.general.total_training_episodes)
    learner.test()
    learner.dump_models(os.path.join(os.getcwd(), "models"))
