# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

from ac import get_ac_policy

cim_path = os.path.dirname(os.path.realpath(__file__))
if cim_path not in sys.path:
    sys.path.insert(0, cim_path)

update_trigger = {"POLICY": 128}
warmup = {"POLICY": 1}

# use agent IDs as policy names since each agent uses a separate policy
rl_policy_func_index = {"POLICY": get_ac_policy}
agent2policy = {"AGENT": "POLICY"}
