# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import cast

from maro.simulator import Env

from tests.rl.gym_wrapper.simulator.business_engine import GymBusinessEngine

env_conf = {
    "topology": "Walker2d-v4",  # HalfCheetah-v4, Hopper-v4, Walker2d-v4, Swimmer-v4, Ant-v4
    "start_tick": 0,
    "durations": 1000,
    "options": {
        "random_seed": None,
    },
}

learn_env = Env(business_engine_cls=GymBusinessEngine, **env_conf)
test_env = Env(business_engine_cls=GymBusinessEngine, **env_conf)
num_agents = len(learn_env.agent_idx_list)

gym_env = cast(GymBusinessEngine, learn_env.business_engine).gym_env
gym_state_dim = gym_env.observation_space.shape[0]
gym_action_dim = gym_env.action_space.shape[0]
action_lower_bound, action_upper_bound = gym_env.action_space.low, gym_env.action_space.high
action_limit = gym_env.action_space.high[0]
