# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import partial

from .config import algorithm, action_shaping_conf, state_dim

action_num = len(action_shaping_conf["action_space"])

if algorithm == "ac":
    from .algorithms.ac import get_ac, get_policy
    policy_creator = {f"ac_{i}.policy": partial(get_policy, state_dim, action_num) for i in range(4)}
    trainer_creator = {f"ac_{i}": partial(get_ac, state_dim) for i in range(4)}
elif algorithm == "ppo":
    from .algorithms.ppo import get_ppo, get_policy
    policy_creator = {f"ppo_{i}.policy": partial(get_policy, state_dim, action_num) for i in range(4)}
    trainer_creator = {f"ppo_{i}": partial(get_ppo, state_dim) for i in range(4)}
elif algorithm == "dqn":
    from .algorithms.dqn import get_dqn, get_policy
    policy_creator = {f"dqn_{i}.policy": partial(get_policy, state_dim, action_num) for i in range(4)}
    trainer_creator = {f"dqn_{i}": get_dqn for i in range(4)}
elif algorithm == "discrete_maddpg":
    from .algorithms.maddpg import get_policy, get_maddpg
    policy_creator = {f"discrete_maddpg_{i}.policy": partial(get_policy, state_dim, action_num) for i in range(4)}
    trainer_creator = {f"discrete_maddpg_{i}": partial(get_maddpg, state_dim, [1]) for i in range(4)}
else:
    raise ValueError(f"Unsupported algorithm: {algorithm}")
