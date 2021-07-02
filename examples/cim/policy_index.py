# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

cim_path = os.path.dirname(os.path.realpath(__file__))
if cim_path not in sys.path:
    sys.path.insert(0, cim_path)
from dqn import get_dqn_policy_for_rollout, get_dqn_policy_for_training
from env_wrapper import AGENT_IDS

# use agent IDs as policy names since each agent uses a separate policy
create_train_policy_func = {name: get_dqn_policy_for_training for name in AGENT_IDS}
create_rollout_policy_func = {name: get_dqn_policy_for_rollout for name in AGENT_IDS}
