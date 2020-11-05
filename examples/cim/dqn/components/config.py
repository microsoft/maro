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
    raw_config = yaml.safe_load(in_file)
    config = convert_dottable(raw_config)
