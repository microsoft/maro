# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys


cim_path = os.path.dirname(os.path.realpath(__file__))
if cim_path not in sys.path:
    sys.path.insert(0, cim_path)
from dqn import get_dqn_policy
from env_wrapper import AGENT_IDS

update_trigger = {name: 128 for name in AGENT_IDS}
warmup = {name: 1 for name in AGENT_IDS}

# use agent IDs as policy names since each agent uses a separate policy
rl_policy_func_index = {name: get_dqn_policy for name in AGENT_IDS}
agent2policy = {name: name for name in AGENT_IDS}
