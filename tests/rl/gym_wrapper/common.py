# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import cast

from gym import spaces

from maro.simulator import Env

from tests.rl.gym_wrapper.simulator.business_engine import GymBusinessEngine

env_conf = {
    # Envs with discrete action: {CartPole-v1}
    # Envs with continuous action: {HalfCheetah-v4, Hopper-v4, Walker2d-v4, Swimmer-v4, Ant-v4}
    "topology": "Walker2d-v4",
    "start_tick": 0,
    "durations": 100000,  # Set a very large number
    "options": {},
}

learn_env = Env(business_engine_cls=GymBusinessEngine, **env_conf)
test_env = Env(business_engine_cls=GymBusinessEngine, **env_conf)
num_agents = len(learn_env.agent_idx_list)

gym_env = cast(GymBusinessEngine, learn_env.business_engine).gym_env
gym_state_dim = gym_env.observation_space.shape[0]
gym_action_space = gym_env.action_space
is_discrete = isinstance(gym_action_space, spaces.Discrete)
if is_discrete:
    gym_action_space = cast(spaces.Discrete, gym_action_space)
    gym_action_dim = 1
    gym_action_num = gym_action_space.n
    action_lower_bound, action_upper_bound = None, None  # Should never be used
    action_limit = None  # Should never be used
else:
    gym_action_space = cast(spaces.Box, gym_action_space)
    gym_action_dim = gym_action_space.shape[0]
    gym_action_num = -1  # Should never be used
    action_lower_bound, action_upper_bound = gym_action_space.low, gym_action_space.high
    action_limit = action_upper_bound[0]
