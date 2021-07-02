# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os.path import dirname, realpath

from maro.rl.learning import SimpleLearner

template_dir = dirname(dirname((realpath(__file__))))
if template_dir not in sys.path:
    sys.path.insert(0, template_dir)

from general import config, get_agent_wrapper, get_env_wrapper, log_dir


if __name__ == "__main__":
    SimpleLearner(
        get_env_wrapper(),
        get_agent_wrapper(),
        num_episodes=config["num_episodes"],
        num_steps=config["num_steps"],
        eval_schedule=config["eval_schedule"],
        log_env_summary=config["log_env_summary"],
        log_dir=log_dir
    ).run()
