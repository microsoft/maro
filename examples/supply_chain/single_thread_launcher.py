# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os.path import dirname, realpath

from maro.rl import Learner, MultiAgentPolicy
from maro.simulator import Env

sc_code_dir = dirname(realpath(__file__))
sys.path.insert(0, sc_code_dir)
from config import config
from env_wrapper import SCEnvWrapper
from exploration import exploration_dict, agent_to_exploration
from policies import policy_dict, agent_to_policy


# Single-threaded launcher
if __name__ == "__main__":
    env = SCEnvWrapper(Env(**config["env"]))
    policy = MultiAgentPolicy(
        policy_dict,
        agent_to_policy,
        exploration_dict=exploration_dict,
        agent_to_exploration=agent_to_exploration    
    )

    # create a learner to start training
    learner = Learner(
        policy, env, config["num_episodes"],
        policy_update_interval=config["policy_update_interval"],
        eval_points=config["eval_points"],
        log_env_metrics=config["log_env_metrics"]
    )
    learner.run()
