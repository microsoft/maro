# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This file is used to load config and convert it into a dotted dictionary.
"""

import io
import os

import yaml


def set_input_dim(config):
    # obtain model input dimension from state shaping configurations
    look_back = config["state_shaping"]["look_back"]
    max_ports_downstream = config["state_shaping"]["max_ports_downstream"]
    num_port_attributes = len(config["state_shaping"]["port_attributes"])
    num_vessel_attributes = len(config["state_shaping"]["vessel_attributes"])

    input_dim = (look_back + 1) * (max_ports_downstream + 1) * num_port_attributes + num_vessel_attributes
    config["agents"]["algorithm"]["input_dim"] = input_dim

    return config


CONFIG_PATH = os.path.join(os.path.split(os.path.realpath(__file__))[0], "../config.yml")
with io.open(CONFIG_PATH, "r") as in_file:
    config = yaml.safe_load(in_file)

DISTRIBUTED_CONFIG_PATH = os.path.join(os.path.split(os.path.realpath(__file__))[0], "../distributed_config.yml")
with io.open(DISTRIBUTED_CONFIG_PATH, "r") as in_file:
    distributed_config = yaml.safe_load(in_file)
