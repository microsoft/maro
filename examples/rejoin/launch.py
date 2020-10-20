# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This script is used to debug distributed algorithm in single host multi-process mode.
"""

import os
import io
import yaml

from maro.utils import convert_dottable

CONFIG_PATH = os.path.join(os.path.split(os.path.realpath(__file__))[0], "config.yml")

with io.open(CONFIG_PATH, "r") as in_file:
    raw_config = yaml.safe_load(in_file)
    config = convert_dottable(raw_config)


ACTOR_NUM = config.learner.peer["actor"]  # must be same as in config
LEARNER_NUM = config.actor.peer["learner"]

learner_path = f"{os.path.split(os.path.realpath(__file__))[0]}/learner.py &"
actor_path = f"{os.path.split(os.path.realpath(__file__))[0]}/actor.py &"

for l_num in range(LEARNER_NUM):
    os.system(f"LOG_LEVEL=DEBUG python3 " + learner_path)

for a_num in range(ACTOR_NUM):
    os.system(f"LOG_LEVEL=DEBUG python3 " + actor_path)
