# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import io
import os
import yaml

import numpy as np

from maro.rl import MultiAgentWrapper, TwoPhaseLinearParameterScheduler
from maro.simulator import Env
from maro.utils import LogFormat, Logger, set_seeds

from examples.cim.dqn.components import CIMActionShaper, CIMStateShaper, CIMExperienceShaper, create_dqn_agent
from examples.cim.dqn.training.actor import BasicRolloutExecutor


def launch(config):
    logger = Logger("CIM-DQN", format_=LogFormat.simple, auto_timestamp=False)
    # Step 1: Initialize a CIM environment for using a toy dataset.
    env = Env(config.env.scenario, config.env.topology, durations=config.env.durations)
    action_space = list(np.linspace(-1.0, 1.0, config.agent.model.output_dim))

    # Step 2: Create state, action and experience shapers. We also need to create an explorer here due to the
    # greedy nature of the DQN algorithm.
    state_shaper = CIMStateShaper(**config.env.state_shaping)
    action_shaper = CIMActionShaper(action_space)
    experience_shaper = CIMExperienceShaper(**config.env.experience_shaping)

    # Step 3: Create agents.
    config.agent.model.input_dim = state_shaper.dim
    set_seeds(config.agent.seed)
    agent = MultiAgentWrapper({name: create_dqn_agent(config.agent) for name in env.agent_idx_list})

    # Step 4: training loop.
    executor = BasicRolloutExecutor(env, agent, state_shaper, action_shaper, experience_shaper)
    scheduler = TwoPhaseLinearParameterScheduler(config.training.max_episode, **config.training.exploration)
    for exploration_params in scheduler:
        exp_by_agent = executor.roll_out(scheduler.iter, exploration_params=exploration_params)
        logger.info(f"ep {scheduler.iter} - performance: {env.metrics}, exploration_params: {exploration_params}")
        for agent_id, exp in exp_by_agent.items():
            exp.update({"loss": [1e8] * len(list(exp.values())[0])})
            agent[agent_id].store_experiences(exp)

        for dqn in agent.agent_dict.values():
            dqn.train()


if __name__ == "__main__":
    from examples.cim.dqn.config import config
    # multi-process mode
    if config.multi_process.enable:
        learner_path = f"{os.path.split(os.path.realpath(__file__))[0]}/training/learner.py &"
        actor_path = f"{os.path.split(os.path.realpath(__file__))[0]}/training/actor.py &"

        # Launch the actor processes
        for _ in range(config.multi_process.num_actors):
            os.system(f"python {actor_path}")

        # Launch the learner process
        os.system(f"python {learner_path}")
    else:
        launch(config)
