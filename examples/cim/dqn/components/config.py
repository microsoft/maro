# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This file is used to load config and convert it into a dotted dictionary.
"""

import io
import os
import yaml

from maro.utils import convert_dottable


CONFIG_PATH = os.path.join(os.path.split(os.path.realpath(__file__))[0], "../config.yml")
with io.open(CONFIG_PATH, "r") as in_file:
    config = yaml.safe_load(in_file)

# obtain model input dimension from state shaping configurations
look_back = config["state_shaping"]["look_back"]
max_ports_downstream = config["state_shaping"]["max_port_downstream"]
num_port_attributes = len(config["state_shaping"]["port_attributes"])
num_vessel_attributes = len(config["state_shaping"]["vessel_attributes"])

input_dim = (look_back + 1) * (max_ports_downstream + 1) * num_port_attributes + num_vessel_attributes
config["agents"]["algorithm"]["input_dim"] = input_dim
config = convert_dottable(config)
