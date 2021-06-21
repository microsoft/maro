# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

from maro.rl import EpsilonGreedyExploration, MultiPhaseLinearExplorationScheduler, Learner, LocalLearner
from maro.simulator import Env

dqn_path = os.path.dirname(os.path.realpath(__file__))  # DQN directory
cim_path = os.path.dirname(dqn_path)  # CIM example directory
sys.path.insert(0, cim_path)
sys.path.insert(0, dqn_path)
from env_wrapper import CIMEnvWrapper
from general import NUM_ACTIONS, config, log_dir
from policy import get_independent_policy_for_training
from rollout_manager import rollout_manager
from policy_manager import policy_manager


if __name__ == "__main__":
    if config["mode"] == "local":
        env = Env(**config["env"]["basic"])
        epsilon_greedy = EpsilonGreedyExploration(num_actions=NUM_ACTIONS)
        epsilon_greedy.register_schedule(
            scheduler_cls=MultiPhaseLinearExplorationScheduler,
            param_name="epsilon",
            last_ep=config["num_episodes"],
            **config["exploration"]
        )
        local_learner = LocalLearner(
            env=CIMEnvWrapper(env, **config["env"]["wrapper"]),
            policies=[get_independent_policy_for_training(config["policy"], i) for i in env.agent_idx_list],
            agent2policy={i: i for i in env.agent_idx_list},
            num_episodes=config["num_episodes"],
            num_steps=config["num_steps"],
            exploration_dict={f"EpsilonGreedy1": epsilon_greedy},
            agent2exploration={i: f"EpsilonGreedy1" for i in env.agent_idx_list},
            log_dir=log_dir
        )
        local_learner.run()
    elif config["mode"] == "multi-process":
        learner = Learner(
            policy_manager=policy_manager,
            rollout_manager=rollout_manager,
            num_episodes=config["num_episodes"],
            # eval_schedule=config["eval_schedule"],
            log_dir=log_dir
        )
        learner.run()
    else:
        print("Two modes are supported: local or multi-process.")
