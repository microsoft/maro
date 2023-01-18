# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import cast

from maro.rl.rl_component.rl_component_bundle import RLComponentBundle
from maro.simulator import Env
from tests.rl.gym_wrapper.simulator.business_engine import GymBusinessEngine
from .config import algorithm

from .env_sampler import GymEnvSampler

env_conf = {
    "business_engine_cls": GymBusinessEngine,
    "topology": "Walker2d-v4",
    "start_tick": 0,
    "durations": 5000,
    "options": {
        "random_seed": None,
    },
}

learn_env = Env(**env_conf)
test_env = learn_env
num_agents = len(learn_env.agent_idx_list)

gym_env = cast(GymBusinessEngine, learn_env.business_engine).gym_env
gym_state_dim = gym_env.observation_space.shape[0]
gym_action_dim = gym_env.action_space.shape[0]
action_lower_bound = [float("-inf") for _ in range(gym_env.action_space.shape[0])]
action_upper_bound = [float("inf") for _ in range(gym_env.action_space.shape[0])]

agent2policy = {agent: f"{algorithm}_{agent}.policy" for agent in learn_env.agent_idx_list}

if algorithm == "ppo":
    from tests.rl.algorithms.ppo import get_policy, get_ppo

    policies = [
        get_policy(f"{algorithm}_{i}.policy", action_lower_bound, action_upper_bound, gym_state_dim, gym_action_dim)
        for i in range(num_agents)
    ]
    trainers = [get_ppo(f"{algorithm}_{i}", gym_state_dim) for i in range(num_agents)]
else:
    raise ValueError(f"Unsupported algorithm: {algorithm}")

rl_component_bundle = RLComponentBundle(
    env_sampler=GymEnvSampler(
        learn_env=learn_env,
        test_env=test_env,
        policies=policies,
        agent2policy=agent2policy,
    ),
    agent2policy=agent2policy,
    policies=policies,
    trainers=trainers,
)