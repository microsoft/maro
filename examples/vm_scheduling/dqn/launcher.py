# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import io
import os
import random
import timeit
import yaml
import numpy as np

from maro.rl import SimpleLearner, TwoPhaseLinearParameterScheduler
from maro.simulator import Env
from maro.utils import LogFormat, Logger, convert_dottable

from components import VMActionShaper, VMStateShaper, DQNAgentManager, TruncatedExperienceShaper, create_dqn_agents
from vm_rl import VMActor


CONFIG_PATH = os.path.join(os.path.split(os.path.realpath(__file__))[0], "experiment/configs/combine_net/config.yml")
with io.open(CONFIG_PATH, "r") as in_file:
    raw_config = yaml.safe_load(in_file)
    config = convert_dottable(raw_config)


if __name__ == "__main__":
    start_time = timeit.default_timer()
    # Step 1: Initialize a VM environment.
    env = Env(
        scenario=config.env.scenario,
        topology=config.env.topology,
        start_tick=config.env.start_tick,
        durations=config.env.durations,
        snapshot_resolution=config.env.resolution
    )

    if config.env.seed is not None:
        env.set_seed(config.env.seed)
        random.seed(config.env.seed)

    # Step 2: Create state, action and experience shapers. We also need to create an explorer here due to the
    # greedy nature of the DQN algorithm.
    state_shaper = VMStateShaper(**config.env.state_shaping)
    action_shaper = VMActionShaper()
    experience_shaper = TruncatedExperienceShaper()

    # Step 3: Create agents and an agent manager.
    agent_name = "allocator"
    # config.agents.model.input_dim = state_shaper.dim
    agent_manager = DQNAgentManager(
        create_dqn_agents(agent_name, config.agents),
        state_shaper=state_shaper,
        action_shaper=action_shaper,
        experience_shaper=experience_shaper
    )

    # Step 4: Create an actor and a learner to start the training process.
    scheduler = TwoPhaseLinearParameterScheduler(config.main_loop.max_episode, **config.main_loop.exploration)
    actor = VMActor(env, agent_manager)
    learner = SimpleLearner(
        agent_manager, actor, scheduler,
        logger=Logger("vm_learner", format_=LogFormat.simple, auto_timestamp=False)
    )
    learner.learn()
    learner.test()
    learner.dump_models(os.path.join(os.getcwd(), "models"))