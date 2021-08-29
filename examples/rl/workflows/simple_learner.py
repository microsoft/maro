# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import yaml
from os.path import dirname, join, realpath

from maro.rl.learning import simple_learner

workflow_dir = dirname((realpath(__file__)))
if workflow_dir not in sys.path:
    sys.path.insert(0, workflow_dir)

with open(join(workflow_dir, "config.yml"), "r") as fp:
    config = yaml.safe_load(fp)

from general import get_env_sampler, get_eval_env_wrapper, log_dir, post_collect, post_evaluate
from rollout import get_agent_wrapper


if __name__ == "__main__":
    simple_learner(
        get_env_sampler(),
        get_agent_wrapper(rollout_only=False),
        num_episodes=config["num_episodes"],
        num_steps=config["num_steps"],
        get_eval_env_wrapper=get_eval_env_wrapper,
        eval_schedule=config["eval_schedule"],
        post_collect=post_collect,
        post_evaluate=post_evaluate,
        log_dir=log_dir
    )
