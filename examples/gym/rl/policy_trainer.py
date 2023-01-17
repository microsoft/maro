# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import partial

from maro.rl.training.utils import extract_trainer_name
from maro.simulator.scenarios.gym.business_engine import GymBusinessEngine

from .config import algorithm
from .env_helper import helper_env

be = helper_env.business_engine
assert isinstance(be, GymBusinessEngine)

if algorithm == "random":
    from .algorithms.random import get_policy

    policy_creator = {
        f"{algorithm}_{i}.policy": partial(get_policy, be.gym_env.action_space) for i in helper_env.agent_idx_list
    }
    trainer_creator = {}
elif algorithm == "ppo":
    from .algorithms.ppo import get_policy, get_ppo

    policy_creator = {f"{algorithm}_{i}.policy": get_policy for i in helper_env.agent_idx_list}
    trainer_creator = {extract_trainer_name(policy_name): get_ppo for policy_name in policy_creator}
elif algorithm == "ddpg":
    from .algorithms.ddpg import get_ddpg, get_policy

    policy_creator = {f"{algorithm}_{i}.policy": get_policy for i in helper_env.agent_idx_list}
    trainer_creator = {extract_trainer_name(policy_name): get_ddpg for policy_name in policy_creator}
elif algorithm == "sac":
    from .algorithms.sac import get_policy, get_sac

    policy_creator = {f"{algorithm}_{i}.policy": get_policy for i in helper_env.agent_idx_list}
    trainer_creator = {extract_trainer_name(policy_name): get_sac for policy_name in policy_creator}
else:
    raise ValueError(f"Unsupported algorithm: {algorithm}")

device_mapping = {policy_name: "cpu" for policy_name in policy_creator}
