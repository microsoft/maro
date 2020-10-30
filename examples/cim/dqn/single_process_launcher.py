# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import numpy as np

from maro.simulator import Env

from maro.rl import AgentManagerMode, MaxDeltaEarlyStoppingChecker, KStepExperienceShaper, SimpleLearner, SimpleActor, \
    TwoPhaseLinearExplorer
from maro.utils import Logger, convert_dottable

from components.action_shaper import CIMActionShaper
from components.agent_manager import create_dqn_agents, DQNAgentManager
from components.config import set_input_dim
from components.experience_shaper import TruncatedExperienceShaper
from components.state_shaper import CIMStateShaper


def launch(config):
    # First determine the input dimension and add it to the config.
    set_input_dim(config)
    config = convert_dottable(config)
    # Step 1: Initialize a CIM environment for using a toy dataset.
    env = Env(config.env.scenario, config.env.topology, durations=config.env.durations)
    agent_id_list = [str(agent_id) for agent_id in env.agent_idx_list]

    # Step 2: Create state, action and experience shapers. We also need to create an explorer here due to the
    # greedy nature of the DQN algorithm.
    state_shaper = CIMStateShaper(**config.state_shaping)
    action_shaper = CIMActionShaper(action_space=list(np.linspace(-1.0, 1.0, config.agents.algorithm.num_actions)))
    if config.experience_shaping.type == "truncated":
        experience_shaper = TruncatedExperienceShaper(**config.experience_shaping.truncated)
    else:
        experience_shaper = KStepExperienceShaper(
            reward_func=lambda mt: 1-mt["container_shortage"]/mt["order_requirements"],
            **config.experience_shaping.k_step
        )

    # Step 3: Create agents and an agent manager.
    agent_manager = DQNAgentManager(
        name="cim_learner",
        mode=AgentManagerMode.TRAIN_INFERENCE,
        agent_dict=create_dqn_agents(agent_id_list, config.agents),
        state_shaper=state_shaper,
        action_shaper=action_shaper,
        experience_shaper=experience_shaper
    )

    # Step 4: Create an actor and a learner to start the training process.
    early_stopping_checker = MaxDeltaEarlyStoppingChecker(
        last_k=config.general.early_stopping.last_k,
        threshold=config.general.early_stopping.threshold
    )
    actor = SimpleActor(env=env, inference_agents=agent_manager)
    learner = SimpleLearner(
        trainable_agents=agent_manager,
        actor=actor,
        explorer=TwoPhaseLinearExplorer(**config.exploration),
        logger=Logger("single_host_cim_learner", auto_timestamp=False)
    )
    learner.train(
        max_episode=config.general.max_episode,
        early_stopping_checker=early_stopping_checker,
        warmup_ep=config.general.early_stopping.warmup_ep,
        early_stopping_metric_func=lambda x: 1 - x["container_shortage"] / x["order_requirements"],
    )
    learner.test()
    learner.dump_models(os.path.join(os.getcwd(), "models"))
    learner.exit()


if __name__ == "__main__":
    from components.config import config
    launch(config)
