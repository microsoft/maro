# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import io
import os
import shutil
import random
import timeit
import yaml
import numpy as np

from maro.rl import TwoPhaseLinearParameterScheduler
from maro.simulator import Env
from maro.utils import LogFormat, Logger, convert_dottable

from components import VMActionShaper, VMStateShaper, DQNAgentManager, TruncatedExperienceShaper, create_dqn_agents
from vm_rl import VMActor, VMLearner


FILE_PATH = os.path.split(os.path.realpath(__file__))[0]
CONFIG_PATH = os.path.join(FILE_PATH, "toy_config.yml")
with io.open(CONFIG_PATH, "r") as in_file:
    raw_config = yaml.safe_load(in_file)
    config = convert_dottable(raw_config)

LOG_PATH = os.path.join(FILE_PATH, "log", config.experiment_name)
if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)
simulation_logger = Logger(tag="simulation", format_=LogFormat.none, dump_folder=LOG_PATH, dump_mode="w", auto_timestamp=False)
test_simulation_logger = Logger(tag="test_simulation", format_=LogFormat.none, dump_folder=LOG_PATH, dump_mode="w", auto_timestamp=False)
dqn_logger = Logger(tag="dqn", format_=LogFormat.none, dump_folder=LOG_PATH, dump_mode="w", auto_timestamp=False)
test_dqn_logger = Logger(tag="test_dqn", format_=LogFormat.none, dump_folder=LOG_PATH, dump_mode="w", auto_timestamp=False)


MODEL_PATH = os.path.join(FILE_PATH, "log", config.experiment_name, "models")
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)


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

    shutil.copy(
        os.path.join(env._business_engine._config_path, "config.yml"),
        os.path.join(LOG_PATH, "BEconfig.yml")
    )
    shutil.copy(CONFIG_PATH, os.path.join(LOG_PATH, "config.yml"))

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
    learner = VMLearner(
        model_path=MODEL_PATH,
        eval_interval=config.eval_interval,
        agent_manager=agent_manager, 
        actor=actor, scheduler=scheduler,
        simulation_logger=simulation_logger,
        test_simulation_logger=test_simulation_logger,
        dqn_logger=dqn_logger,
        test_dqn_logger=test_dqn_logger
    )
    learner.learn()
    learner.test()
