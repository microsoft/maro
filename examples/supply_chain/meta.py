# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

from maro.rl import NullPolicy

cim_path = os.path.dirname(__file__)
sys.path.insert(0, cim_path)
from dqn import get_dqn_policy_for_rollout, get_dqn_policy_for_training
from or_policies import (
    get_consumer_baseline_policy, get_consumer_eoq_policy, get_consumer_minmax_policy, get_producer_baseline_policy
)

SC_CREATE_ROLLOUT_POLICY_FUNC = {
    "consumer": get_consumer_minmax_policy,
    "consumerstore": get_dqn_policy_for_rollout, 
    "producer": get_producer_baseline_policy,
    "facility": lambda: NullPolicy(),
    "product": lambda: NullPolicy(),
    "productstore": lambda: NullPolicy()
}

SC_CREATE_TRAIN_POLICY_FUNC = {"consumerstore": get_dqn_policy_for_training}
