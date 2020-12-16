# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from statistics import mean

import numpy as np


from maro.rl import (
    AgentManagerMode, KStepExperienceShaper, Scheduler, SimpleActor, SimpleLearner,
    StaticExplorationParameterGenerator, TwoPhaseLinearExplorationParameterGenerator
)
from maro.simulator import Env
from maro.utils import Logger, convert_dottable

from components import (
    CIMActionShaper, CIMStateShaper, DQNAgentManager, TruncatedExperienceShaper, create_dqn_agents, set_input_dim
)


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
        experience_shaper=experience_shaper
    )

    # Step 4: Create an actor and a learner to start the training process.
    def early_stopping_callback(perf_history):
        last_k = config.main_loop.early_stopping.last_k
        perf_threshold = config.main_loop.early_stopping.perf_threshold
        perf_stability_threshold = config.main_loop.early_stopping.perf_stability_threshold
        if len(perf_history) < last_k:
            return False

        metric_series = list(
            map(lambda p: 1 - p["container_shortage"] / p["order_requirements"], perf_history[-last_k:])
        )
        mean_perf = mean(metric_series)
        max_delta = max(abs(metric_series[i] - metric_series[i - 1]) / metric_series[i - 1] for i in range(1, last_k))
        return mean_perf > perf_threshold and max_delta < perf_stability_threshold

    scheduler = Scheduler(
        config.main_loop.max_episode,
        warmup_ep=config.main_loop.early_stopping.warmup_ep,
        early_stopping_callback=early_stopping_callback,
        exploration_parameter_generator_cls=TwoPhaseLinearExplorationParameterGenerator,
        exploration_parameter_generator_config=config.main_loop.exploration,
        logger=Logger("single_host_cim_learner", auto_timestamp=False)
    )

    actor = SimpleActor(env, agent_manager)
    learner = SimpleLearner(agent_manager, actor, scheduler)
    learner.learn()
    learner.test()
    learner.dump_models(os.path.join(os.getcwd(), "models"))


if __name__ == "__main__":
    from components.config import config
    launch(config)
