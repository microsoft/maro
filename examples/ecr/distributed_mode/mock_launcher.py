# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

GROUP = 'ecr_0'
ACTOR_NUM = 3
LEARNER_NUM = 1

learner_path = "learner_launcher.py &"
actor_path = "actor_launcher.py &"

for l_num in range(LEARNER_NUM):
    os.system(f"GROUP={GROUP} python " + learner_path)

for a_num in range(ACTOR_NUM):
    os.system(f"GROUP={GROUP} python " + actor_path)