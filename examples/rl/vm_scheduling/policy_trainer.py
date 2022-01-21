# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import partial

from .config import algorithm, state_dim, num_pms

action_num = num_pms + 1  # action could be any PM or postponement, hence the plus 1

if algorithm == "ac":
    from .algorithms.ac import get_ac, get_discrete_policy_gradient
    policy_creator = {
        f"ac_{i}.policy": partial(get_discrete_policy_gradient, state_dim=state_dim, action_num=action_num)
        for i in range(4)
    }
    trainer_creator = {f"ac_{i}": partial(get_ac, state_dim=state_dim) for i in range(4)}
elif algorithm == "dqn":
    from .algorithms.dqn import get_dqn, get_value_based_policy
    policy_creator = {
        f"dqn_{i}.policy": partial(get_value_based_policy, state_dim=state_dim, action_num=action_num)
        for i in range(4)
    }
    trainer_creator = {f"dqn_{i}": get_dqn for i in range(4)}
else:
    raise ValueError(f"Unsupported algorithm: {algorithm}")
