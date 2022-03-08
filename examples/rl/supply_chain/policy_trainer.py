# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import partial

from .config import rl_algorithm, NUM_CONSUMER_ACTIONS
from .env_helper import state_dim

if rl_algorithm == "dqn":
    from .algorithms.dqn import get_dqn, get_policy
    policy_creator = {f"dqn_{i}.policy": partial(get_policy, state_dim, NUM_CONSUMER_ACTIONS) for i in range(4)}
    trainer_creator = {f"dqn_{i}": get_dqn for i in range(4)}
else:
    raise ValueError(f"Unsupported algorithm: {rl_algorithm}")
