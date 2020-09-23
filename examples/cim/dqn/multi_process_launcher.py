# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This script is used to debug distributed algorithm in single host multi-process mode.
"""

import os

ACTOR_NUM = 1   # must be same as in config
LEARNER_NUM = 1

learner_path = "components/dist_learner.py &"
actor_path = "components/dist_actor.py &"

for l_num in range(LEARNER_NUM):
    os.system(f"python " + learner_path)

for a_num in range(ACTOR_NUM):
    os.system(f"python " + actor_path)
