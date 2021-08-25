# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

cim_path = os.path.dirname(os.path.realpath(__file__))
if cim_path not in sys.path:
    sys.path.insert(0, cim_path)
from ac import get_ac_policy
from dqn import get_dqn_policy

rl_policy_func_index = {
    "dqn": get_dqn_policy,
    "ac.0": get_ac_policy,
    "ac.1": get_ac_policy,
    "ac.2": get_ac_policy
}

agent2policy = {
    0: "ac.0",
    1: "ac.1",
    2: "dqn",
    3: "ac.2"
}
