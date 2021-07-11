# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import importlib
import sys
import yaml
from os.path import dirname, join, realpath

workflow_dir = dirname(realpath(__file__))
rl_example_dir = dirname(workflow_dir)

if rl_example_dir not in sys.path:
    sys.path.insert(0, rl_example_dir)

config_path = join(workflow_dir, "config.yml")
with open(config_path, "r") as config_file:
    config = yaml.safe_load(config_file)

log_dir = join(rl_example_dir, "log", config["job_name"])

module = importlib.import_module(f"{config['scenario']}")

agent2policy = getattr(module, "agent2policy")
get_env_wrapper = getattr(module, "get_env_wrapper")
policy_func_index = getattr(module, "policy_func_index")
update_trigger = getattr(module, "update_trigger")
warmup = getattr(module, "warmup")
