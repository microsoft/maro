# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os.path import dirname, realpath

from maro.rl.learning import AgentWrapper, SimpleLearner


workflow_dir = dirname((realpath(__file__)))
if workflow_dir not in sys.path:
    sys.path.insert(0, workflow_dir)

from general import (
    agent2exploration, agent2policy, config, policy_func_index, exploration_func_index, get_env_wrapper, log_dir
)


if __name__ == "__main__":
    SimpleLearner(
        get_env_wrapper(),
        AgentWrapper(
            {name: func(learning=False) for name, func in policy_func_index.items()},
            agent2policy,
            exploration_dict={name: func() for name, func in exploration_func_index.items()},
            agent2exploration=agent2exploration
        ),
        num_episodes=config["num_episodes"],
        num_steps=config["num_steps"],
        eval_schedule=config["eval_schedule"],
        log_env_summary=config["log_env_summary"],
        log_dir=log_dir
    ).run()
