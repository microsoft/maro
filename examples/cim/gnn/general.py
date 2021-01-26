# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This file is used to load config and convert it into a dotted dictionary.
"""

import datetime
import io
import os

import yaml

from maro.utils import Logger, convert_dottable


config_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], "../config.yml")
with io.open(config_path, "r") as in_file:
    config = yaml.safe_load(in_file)

config = convert_dottable(config)

DISTRIBUTED_CONFIG_PATH = os.path.join(os.path.split(os.path.realpath(__file__))[0], "../distributed_config.yml")
with io.open(DISTRIBUTED_CONFIG_PATH, "r") as in_file:
    distributed_config = yaml.safe_load(in_file)

distributed_config = convert_dottable(distributed_config)

# Generate log path.
date_str = datetime.datetime.now().strftime("%Y%m%d")
time_str = datetime.datetime.now().strftime("%H%M%S.%f")
subfolder_name = f"{config.env.param.topology}_{time_str}"

# Log path.
config.log.path = os.path.join(config.log.path, date_str, subfolder_name)
if not os.path.exists(config.log.path):
    os.makedirs(config.log.path)

simulation_logger = Logger(tag="simulation", dump_folder=config.log.path, auto_timestamp=False)
training_logger = Logger(tag="training", dump_folder=config.log.path, auto_timestamp=False)
