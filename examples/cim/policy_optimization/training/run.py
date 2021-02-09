# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import io
import os
import yaml
from collections import deque
from statistics import mean

import numpy as np

from maro.simulator import Env
from maro.rl import MultiAgentWrapper, Scheduler
from maro.utils import LogFormat, Logger, convert_dottable

from examples.cim.policy_optimization.components import (
    CIMActionShaper, CIMStateShaper, CIMExperienceShaper, create_po_agents
)

from examples.cim.policy_optimization.training.rollout_executor import SimpleRolloutExecutor


def launch(config):
    logger = Logger("CIM-PO", format_=LogFormat.simple, auto_timestamp=False)
    # Step 1: initialize a CIM environment for using a toy dataset.
    env = Env(config.env.scenario, config.env.topology, durations=config.env.durations)

    # Step 2: create state, action and experience shapers. We also need to create an explorer here due to the
    # greedy nature of the DQN algorithm.
    state_shaper = CIMStateShaper(**config.env.state_shaping)
    action_shaper = CIMActionShaper(action_space=list(np.linspace(-1.0, 1.0, config.agent.actor_model.output_dim)))
    experience_shaper = CIMExperienceShaper(**config.env.experience_shaping)

    # Step 3: create an agent manager.
    config.agent.actor_model.input_dim = state_shaper.dim
    config.agent.critic_model.input_dim = state_shaper.dim
    config.agent.names = [str(agent_id) for agent_id in env.agent_idx_list]
    agent = MultiAgentWrapper(create_po_agents(config.agent))

    # Step 4: Create an actor and a learner to start the training process.
    executor = SimpleRolloutExecutor(env, agent, state_shaper, action_shaper, experience_shaper)
    k, warmup_ep, perf_thresh = config.training.k, config.training.warmup_ep, config.training.perf_thresh
    perf_history = deque()
    for ep in range(config.training.max_episode):
        exp_by_agent = executor.roll_out(ep)
        logger.info(f"ep {ep} - performance: {env.metrics}")
        fulfillment = 1 - env.metrics["container_shortage"] / env.metrics["order_requirements"]
        perf_history.append(fulfillment)
        if len(perf_history) > k:
            perf_history.popleft()
        if ep >= warmup_ep and min(perf_history) >= perf_thresh:
            logger.info(f"{k} consecutive fulfillment rates above threshold {perf_thresh}. Training complete")
            break
        # model training
        for agent_id, exp in exp_by_agent.items():
            agent[agent_id].train(exp["state"], exp["action"], exp["log_action_prob"], exp["reward"])


if __name__ == "__main__":
    config_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], "../config.yml")
    with io.open(config_path, "r") as in_file:
        config = convert_dottable(yaml.safe_load(in_file))
    launch(config)
