# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

from maro.rl import (
    EpsilonGreedyExploration, MultiPhaseLinearExplorationScheduler, LocalDecisionGenerator,
    LocalRolloutManager, MultiProcessRolloutManager
)
from maro.simulator import Env

dqn_path = os.path.dirname(os.path.realpath(__file__))  # DQN directory
cim_path = os.path.dirname(dqn_path)  # CIM example directory
sys.path.insert(0, cim_path)
sys.path.insert(0, dqn_path)
from env_wrapper import CIMEnvWrapper
from general import AGENT_IDS, NUM_ACTIONS, config, log_dir
from policy import get_independent_policy_for_rollout


def get_env():
    return CIMEnvWrapper(Env(**config["env"]["basic"]), **config["env"]["wrapper"])

def get_decision_generator():
    epsilon_greedy = EpsilonGreedyExploration(num_actions=NUM_ACTIONS)
    epsilon_greedy.register_schedule(
        scheduler_cls=MultiPhaseLinearExplorationScheduler,
        param_name="epsilon",
        last_ep=config["num_episodes"],
        **config["exploration"]
    )
    return LocalDecisionGenerator(
        agent2policy={i: i for i in AGENT_IDS},
        policies=[get_independent_policy_for_rollout(i) for i in AGENT_IDS],
        exploration_dict={f"EpsilonGreedy": epsilon_greedy},
        agent2exploration={i: "EpsilonGreedy" for i in AGENT_IDS},
        log_dir=log_dir
    )


if config["distributed"]["rollout_mode"] == "local":
    env = Env(**config["env"]["basic"])
    rollout_manager = LocalRolloutManager(
        CIMEnvWrapper(env, **config["env"]["wrapper"]),
        [get_independent_policy_for_rollout(i) for i in env.agent_idx_list],
        {i: i for i in env.agent_idx_list},
        num_steps=config["num_steps"],
        exploration_dict={f"EpsilonGreedy": EpsilonGreedyExploration(num_actions=NUM_ACTIONS)},
        agent2exploration={i: "EpsilonGreedy" for i in env.agent_idx_list},
        log_dir=log_dir
    )
else:
    rollout_manager = MultiProcessRolloutManager(
        config["distributed"]["num_rollout_workers"],
        get_env,
        get_decision_generator,
        num_steps=config["num_steps"],
        log_dir=log_dir,
    )
