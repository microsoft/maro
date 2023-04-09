# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import cast

from maro.simulator import Env

from .simulator.business_engine import GymBusinessEngine

env_conf = {
    "topology": "Walker2d-v4",  # HalfCheetah-v4, Hopper-v4, Walker2d-v4, Swimmer-v4, Ant-v4
    "start_tick": 0,
    "durations": 100000,  # Set a very large number
    "options": {},
}

env = Env(business_engine_cls=GymBusinessEngine, **env_conf)
num_agents = len(env.agent_idx_list)

gym_env = cast(GymBusinessEngine, env.business_engine).gym_env
gym_obs_space = gym_env.observation_space
gym_action_space = gym_env.action_space
gym_state_dim = gym_env.observation_space.shape[0]
gym_action_dim = gym_action_space.shape[0]
obs_lower_bound, obs_upper_bound = gym_obs_space.low, gym_obs_space.high
action_lower_bound, action_upper_bound = gym_action_space.low, gym_action_space.high
action_limit = gym_action_space.high[0]
