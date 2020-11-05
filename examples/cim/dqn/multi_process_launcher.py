# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This script is used to debug distributed algorithm in single host multi-process mode.
"""

import os

from .components.config import config

ACTOR_NUM = config.distributed.learner.peer["actor"]  # must be same as in config
LEARNER_NUM = config.distributed.actor.peer["learner"]

learner_path = f"{os.path.split(os.path.realpath(__file__))[0]}/dist_learner.py &"
actor_path = f"{os.path.split(os.path.realpath(__file__))[0]}/dist_actor.py &"

for l_num in range(LEARNER_NUM):
    os.system(f"python {learner_path}")

for a_num in range(ACTOR_NUM):
    os.system(f"python {actor_path}")
