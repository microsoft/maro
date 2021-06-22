# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os
import sys

from maro.rl import EpsilonGreedyExploration, MultiPhaseLinearExplorationScheduler, actor, policy_server
from maro.simulator import Env

async_mode_path = os.path.dirname(os.path.realpath(__file__))  # DQN async mode directory
dqn_path = os.path.dirname(async_mode_path)  # DQN directory
cim_path = os.path.dirname(dqn_path)  # CIM example directory
sys.path.insert(0, cim_path)
sys.path.insert(0, dqn_path)
sys.path.insert(0, async_mode_path)
from agent_wrapper import get_agent_wrapper
from env_wrapper import CIMEnvWrapper
from general import NUM_ACTIONS, config, log_dir
from policy import get_independent_policy_for_training
from policy_manager import policy_manager


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("i", type=int, choices=[0, 1], help="Identity of process: 0 - policy, 1 - actor")
    args = parser.parse_args()
    if args.i == 0:
        policy_server(
            policy_manager,
            config["distributed"]["num_actors"],
        )
        """
        policy_manager: AbsPolicyManager,
        num_actors: int,
        group: str,
        num_requests_per_inference: int,
        proxy_kwargs: dict = {}
        """
    elif args.i == 1:
        actor(
            actor_id,
            lambda: CIMEnvWrapper(Env(**config["env"]["basic"]), **config["env"]["wrapper"]),
            get_agent_wrapper,
            config["num_episodes"],
            config["distributeed"]["group"],
            num_steps=config["num_steps"],
            log_dir=log_dir,
        )
