# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

from maro.rl.policy import NullPolicy

cim_path = os.path.dirname(__file__)
sys.path.insert(0, cim_path)
from dqn import get_dqn_policy
from or_policies import (
    get_consumer_baseline_policy, get_consumer_eoq_policy, get_consumer_minmax_policy, get_producer_baseline_policy
)

NUM_RL_POLICIES = 100

rollout_policy_func_index = {
    "consumer": get_consumer_minmax_policy,
    "producer": get_producer_baseline_policy,
    "facility": lambda: NullPolicy(),
    "product": lambda: NullPolicy(),
    "productstore": lambda: NullPolicy(),
    # consumer store policies
    **{f"consumerstore-{i}": get_dqn_policy_for_rollout for i in range(NUM_RL_POLICIES)}, 
}

policy_func_index = {f"consumerstore-{i}": get_dqn_policy for i in range(NUM_RL_POLICIES)}
