# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import yaml

from maro.simulator import Env

FILE_PATH = os.path.dirname(os.path.realpath(__file__))

DEFAULT_CONFIG_PATH = os.path.join(FILE_PATH, "config.yml")
with open(os.getenv("CONFIG_PATH", default=DEFAULT_CONFIG_PATH), "r") as config_file:
    config = yaml.safe_load(config_file)

log_dir = os.path.join(FILE_PATH, "logs", config["experiment_name"])

# Obtain model input and output dimensions from env wrapper settings
config["policy"]["model"]["network"]["input_dim"] = (
    (config["env"]["wrapper"]["look_back"] + 1)
    * (config["env"]["wrapper"]["max_ports_downstream"] + 1)
    * len(config["env"]["wrapper"]["port_attributes"])
    + len(config["env"]["wrapper"]["vessel_attributes"])
)
config["policy"]["model"]["network"]["output_dim"] = config["env"]["wrapper"]["num_actions"]

NUM_ACTIONS = config["env"]["wrapper"]["num_actions"]
AGENT_IDS = Env(**config["env"]["basic"]).agent_idx_list
NUM_POLICY_SERVERS = config["multi-process"]["num_policy_servers"]
