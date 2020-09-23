# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This script is used to debug distributed algorithm in single host multi-process mode.
"""

import io
import os
import yaml

from maro.utils import Logger, convert_dottable


config_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], "config.yml")
with io.open(config_path, "r") as in_file:
    raw_config = yaml.safe_load(in_file)
    config = convert_dottable(raw_config)

ACTOR_NUM = config.distributed.learner.peer["actor_worker"]  # must be same as in config
LEARNER_NUM = config.distributed.actor.peer["actor"]

learner_path = f"{os.path.split(os.path.realpath(__file__))[0]}/components/dist_learner.py &"
actor_path = f"{os.path.split(os.path.realpath(__file__))[0]}/components/dist_actor.py &"

for l_num in range(LEARNER_NUM):
    os.system(f"python " + learner_path)

for a_num in range(ACTOR_NUM):
    os.system(f"python " + actor_path)
