# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import partial

from .config import algorithm, state_dim, num_features, num_pms

action_num = num_pms + 1  # action could be any PM or postponement, hence the plus 1

if algorithm == "ac":
    from .algorithms.ac import get_ac, get_policy
    policy_creator = {"ac.policy": partial(get_policy, state_dim, action_num, num_features)}
    trainer_creator = {"ac": partial(get_ac, state_dim, num_features)}
elif algorithm == "dqn":
    from .algorithms.dqn import get_dqn, get_policy
    policy_creator = {"dqn.policy": partial(get_policy, state_dim, action_num, num_features)}
    trainer_creator = {"dqn": get_dqn}
else:
    raise ValueError(f"Unsupported algorithm: {algorithm}")
