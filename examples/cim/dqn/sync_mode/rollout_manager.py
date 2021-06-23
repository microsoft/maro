# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

from maro.rl import EpsilonGreedyExploration, LocalRolloutManager, MultiProcessRolloutManager

from maro.simulator import Env

sync_mode_path = os.path.dirname(os.path.realpath(__file__))  # DQN sync mode directory
dqn_path = os.path.dirname(sync_mode_path)  # DQN directory
cim_path = os.path.dirname(dqn_path)  # CIM example directory
sys.path.insert(0, cim_path)
sys.path.insert(0, dqn_path)
sys.path.insert(0, sync_mode_path)
from agent_wrapper import get_agent_wrapper
from env_wrapper import CIMEnvWrapper
from general import NUM_ACTIONS, config, log_dir
from policy import get_independent_policy_for_rollout


if config["sync"]["rollout_mode"] == "local":
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
        config["sync"]["num_rollout_workers"],
        lambda: CIMEnvWrapper(Env(**config["env"]["basic"]), **config["env"]["wrapper"]),
        get_agent_wrapper,
        num_steps=config["num_steps"],
        log_dir=log_dir,
    )
