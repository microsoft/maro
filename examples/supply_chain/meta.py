# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

from maro.rl import NullPolicy

sc_path = os.path.dirname(os.path.realpath(__file__))
if sc_path not in sys.path:
    sys.path.insert(0, sc_path)
from dqn import get_dqn_policy_for_rollout, get_dqn_policy_for_training
from or_policies import (
    get_consumer_baseline_policy, get_consumer_eoq_policy, get_consumer_minmax_policy, get_producer_baseline_policy
)

create_rollout_policy_func = {
    "consumer": get_consumer_minmax_policy,
    "consumerstore": get_dqn_policy_for_rollout, 
    "producer": get_producer_baseline_policy,
    "facility": lambda: NullPolicy(),
    "product": lambda: NullPolicy(),
    "productstore": lambda: NullPolicy()
}

create_train_policy_func = {"consumerstore": get_dqn_policy_for_training}
