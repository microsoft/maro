# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

from maro.rl.policy import NullPolicy

sc_path = os.path.dirname(__file__)
sys.path.insert(0, sc_path)
from dqn import get_dqn_policy
from env_wrapper import AGENT_IDS
from or_policies import (
    get_consumer_baseline_policy, get_consumer_eoq_policy, get_consumer_minmax_policy, get_producer_baseline_policy
)

NUM_RL_POLICIES = 10

non_rl_policy_func_index = {
    # "consumer": get_consumer_minmax_policy,
    "producer": get_producer_baseline_policy,
    "facility": lambda: NullPolicy(),
    "product": lambda: NullPolicy(),
    "productstore": lambda: NullPolicy()
}

rl_policy_func_index = {f"consumer-{i}": get_dqn_policy for i in range(NUM_RL_POLICIES)}
consumers = [agent_id for agent_id in AGENT_IDS if agent_id.startswith("consumer")]
agent2policy = {
    agent_id: agent_id.split(".")[0] for agent_id in AGENT_IDS if not agent_id.startswith("consumerstore")
}

for i, agent_id in enumerate(consumers):
    agent2policy[agent_id] = f"consumer-{i % NUM_RL_POLICIES}"

update_trigger = {name: 1 for name in rl_policy_func_index}
warmup = {name: 1 for name in rl_policy_func_index}
