# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os.path import dirname, realpath

from maro.rl.learning import SimpleLearner


workflow_dir = dirname((realpath(__file__)))
if workflow_dir not in sys.path:
    sys.path.insert(0, workflow_dir)

from agent_wrapper import get_agent_wrapper
from general import config, get_env_wrapper, log_dir, post_collect, post_evaluate


if __name__ == "__main__":
    SimpleLearner(
        get_env_wrapper(),
        get_agent_wrapper(),
        num_episodes=config["num_episodes"],
        num_steps=config["num_steps"],
        eval_schedule=config["eval_schedule"],
        post_collect=post_collect,
        post_evaluate=post_evaluate,
        log_dir=log_dir
    ).run()
