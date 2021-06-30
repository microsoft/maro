# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
import yaml

example_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, example_dir)

config_path = os.path.join(example_dir, "config.yml")
with open(config_path, "r") as config_file:
    config = yaml.safe_load(config_file)

log_dir = os.path.join(example_dir, "logs", config["experiment_name"])

scenario = config["scenario"]
if scenario == "cim":
    from cim.env_wrapper import get_env_wrapper
    from cim.agent_wrapper import get_agent_wrapper
    from cim.meta import create_rollout_policy_func, create_train_policy_func
else:
    raise ValueError(f"Unsupported scenario: {scenario}. Supported scenarios: 'cim'")
