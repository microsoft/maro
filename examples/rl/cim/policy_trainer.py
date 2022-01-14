# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import partial

from .config import algorithm, action_shaping_conf, state_dim

action_num = len(action_shaping_conf["action_space"])

if algorithm == "ac":
    from .algorithms.ac import get_ac, get_discrete_policy_gradient
    policy_creator = {
        f"ac_{i}.{i}": partial(get_discrete_policy_gradient, state_dim=state_dim, action_num=action_num)
        for i in range(4)
    }
    trainer_creator = {f"ac_{i}": partial(get_ac, state_dim=state_dim) for i in range(4)}
elif algorithm == "dqn":
    from .algorithms.dqn import get_dqn, get_value_based_policy
    policy_creator = {
        f"dqn_{i}.{i}": partial(get_value_based_policy, state_dim=state_dim, action_num=action_num)
        for i in range(4)
    }
    trainer_creator = {f"dqn_{i}": get_dqn for i in range(4)}
elif algorithm == "discrete_maddpg":
    from .algorithms.maddpg import get_discrete_policy_gradient, get_maddpg
    policy_creator = {
        f"discrete_maddpg_{i}.{i}": partial(get_discrete_policy_gradient, state_dim=state_dim, action_num=action_num)
        for i in range(4)
    }
    trainer_creator = {
        f"discrete_maddpg_{i}": partial(get_maddpg, state_dim=state_dim, action_dims=[1]) for i in range(4)
    }
else:
    raise ValueError(f"Unsupported algorithm: {algorithm}")
