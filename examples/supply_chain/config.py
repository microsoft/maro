# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import yaml
from os import getenv
from os.path import dirname, join, realpath

from maro.simulator import Env

sc_code_dir = dirname(realpath(__file__))
sys.path.insert(0, sc_code_dir)
from env_wrapper import SCEnvWrapper


default_config_path = join(dirname(realpath(__file__)), "config.yml")
with open(getenv("CONFIG_PATH", default=default_config_path), "r") as config_file:
    config = yaml.safe_load(config_file)

topology = config["env"]["topology"]
config["env"]["topology"] = join(sc_code_dir, "topologies", topology)
env = SCEnvWrapper(Env(**config["env"]))
consumer_agent_ids = [f"consumer.{info.id}" for info in env.agent_idx_list]
producer_agent_ids = [f"producer.{info.id}" for info in env.agent_idx_list]
config["agent_ids"] = consumer_agent_ids + producer_agent_ids
config["policy"]["consumer"]["model"]["network"]["input_dim"] = env.dim
