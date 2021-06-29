# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

cim_path = os.path.dirname(__file__)
sys.path.insert(0, cim_path)
from dqn import get_dqn_policy_for_rollout, get_dqn_policy_for_training
from env_wrapper import CIM_AGENT_IDS

CIM_POLICY_NAMES = CIM_AGENT_IDS  # use agent IDs as policy names since each agent uses a separate policy
CIM_CREATE_POLICY_FUNC = {name: get_dqn_policy_for_training for name in CIM_POLICY_NAMES}
