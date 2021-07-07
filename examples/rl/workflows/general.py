# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import yaml
from os.path import dirname, join, realpath

workflow_dir = dirname(realpath(__file__))
rl_example_dir = dirname(workflow_dir)

config_path = join(workflow_dir, "config.yml")
with open(config_path, "r") as config_file:
    config = yaml.safe_load(config_file)

log_dir = join(rl_example_dir, "logs", config["job_name"])

scenario_dir = join(rl_example_dir, config["scenario"])
if scenario_dir not in sys.path:
    sys.path.insert(0, scenario_dir)

import get_env_wrapper, get_agent_wrapper, policy_func_index
