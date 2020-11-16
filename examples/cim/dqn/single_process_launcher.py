# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from statistics import mean

import numpy as np

from components.action_shaper import CIMActionShaper
from components.agent_manager import DQNAgentManager, create_dqn_agents
from components.config import set_input_dim
from components.experience_shaper import TruncatedExperienceShaper
from components.state_shaper import CIMStateShaper

from maro.rl import (
    AgentManagerMode, EpsilonGreedyExplorer, KStepExperienceShaper, MaxDeltaEarlyStoppingChecker, SimpleActor,
    SimpleEarlyStoppingChecker, SimpleLearner, two_phase_linear_epsilon_schedule
)
from maro.simulator import Env
from maro.utils import Logger, convert_dottable


def launch(config):
    # First determine the input dimension and add it to the config.
    set_input_dim(config)
    config = convert_dottable(config)
    # Step 1: Initialize a CIM environment for using a toy dataset.
    env = Env(config.env.scenario, config.env.topology, durations=config.env.durations)
    agent_id_list = [str(agent_id) for agent_id in env.agent_idx_list]
    action_space = list(np.linspace(-1.0, 1.0, config.agents.algorithm.num_actions))

    # Step 2: Create state, action and experience shapers. We also need to create an explorer here due to the
    # greedy nature of the DQN algorithm.
    state_shaper = CIMStateShaper(**config.state_shaping)
    action_shaper = CIMActionShaper(action_space=action_space)
    if config.experience_shaping.type == "truncated":
        experience_shaper = TruncatedExperienceShaper(**config.experience_shaping.truncated)
    else:
        experience_shaper = KStepExperienceShaper(
            reward_func=lambda mt: 1 - mt["container_shortage"] / mt["order_requirements"],
            **config.experience_shaping.k_step
        )

    # Step 3: Create agents and an agent manager.
    agent_manager = DQNAgentManager(
        name="cim_learner",
        mode=AgentManagerMode.TRAIN_INFERENCE,
        agent_dict=create_dqn_agents(agent_id_list, config.agents),
        state_shaper=state_shaper,
        action_shaper=action_shaper,
        experience_shaper=experience_shaper,
        explorer=EpsilonGreedyExplorer(config.agents.algorithm.num_actions),
    )

    # Step 4: Create an actor and a learner to start the training process.
    perf_checker = SimpleEarlyStoppingChecker(
        last_k=config.main_loop.early_stopping.last_k,
        threshold=config.main_loop.early_stopping.perf_threshold,
        measure_func=lambda vals: mean(vals)
    )

    perf_stability_checker = MaxDeltaEarlyStoppingChecker(
        last_k=config.main_loop.early_stopping.last_k,
        threshold=config.main_loop.early_stopping.perf_stability_threshold
    )

    combined_checker = perf_checker & perf_stability_checker

    exploration_schedule = two_phase_linear_epsilon_schedule(**config.main_loop.exploration)
    actor = SimpleActor(env, agent_manager)
    learner = SimpleLearner(
        agent_manager=agent_manager,
        actor=actor,
        logger=Logger("single_host_cim_learner", auto_timestamp=False)
    )
    learner.learn(
        exploration_schedule,
        early_stopping_checker=combined_checker,
        warmup_ep=config.main_loop.early_stopping.warmup_ep,
        early_stopping_metric_func=lambda x: 1 - x["container_shortage"] / x["order_requirements"],
    )
    learner.test()
    learner.dump_models(os.path.join(os.getcwd(), "models"))


if __name__ == "__main__":
    from components.config import config
    launch(config)
