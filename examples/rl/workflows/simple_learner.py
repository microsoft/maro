# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import importlib
import sys
import yaml
from os.path import dirname, join, realpath

from maro.rl.learning import simple_learner

workflow_dir = dirname((realpath(__file__)))
if workflow_dir not in sys.path:
    sys.path.insert(0, workflow_dir)

with open(join(workflow_dir, "config.yml"), "r") as fp:
    config = yaml.safe_load(fp)

rl_example_dir = dirname(workflow_dir)
if rl_example_dir not in sys.path:
    sys.path.insert(0, rl_example_dir)

log_dir = join(rl_example_dir, "log", config["job"])

module = importlib.import_module(config["scenario"])
get_env_sampler = getattr(module, "get_env_sampler")
policy_func_dict = getattr(module, "policy_func_dict")
post_collect = getattr(module, "post_collect", None)
post_evaluate = getattr(module, "post_evaluate", None)


if __name__ == "__main__":
    simple_learner(
        get_env_sampler,
        num_episodes=config["num_episodes"],
        num_steps=config["num_steps"],
        eval_schedule=config["eval_schedule"],
        post_collect=post_collect,
        post_evaluate=post_evaluate,
        log_dir=log_dir
    )
