# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import cast

from maro.rl.rl_component.rl_component_bundle import RLComponentBundle
from maro.simulator import Env

from .config import algorithm, env_conf
from .env_sampler import GymEnvSampler
from tests.rl.gym_wrapper.simulator.business_engine import GymBusinessEngine

learn_env = Env(business_engine_cls=GymBusinessEngine, **env_conf)
test_env = learn_env
num_agents = len(learn_env.agent_idx_list)

gym_env = cast(GymBusinessEngine, learn_env.business_engine).gym_env
gym_state_dim = gym_env.observation_space.shape[0]
gym_action_dim = gym_env.action_space.shape[0]
action_lower_bound, action_upper_bound = gym_env.action_space.low, gym_env.action_space.high
action_limit = gym_env.action_space.high[0]

agent2policy = {agent: f"{algorithm}_{agent}.policy" for agent in learn_env.agent_idx_list}

if algorithm == "ac":
    from tests.rl.algorithms.ac import get_ac_policy, get_ac_trainer

    policies = [
        get_ac_policy(f"{algorithm}_{i}.policy", action_lower_bound, action_upper_bound, gym_state_dim, gym_action_dim)
        for i in range(num_agents)
    ]
    trainers = [get_ac_trainer(f"{algorithm}_{i}", gym_state_dim) for i in range(num_agents)]
elif algorithm == "ppo":
    from tests.rl.algorithms.ppo import get_ppo_policy, get_ppo_trainer

    policies = [
        get_ppo_policy(f"{algorithm}_{i}.policy", action_lower_bound, action_upper_bound, gym_state_dim, gym_action_dim)
        for i in range(num_agents)
    ]
    trainers = [get_ppo_trainer(f"{algorithm}_{i}", gym_state_dim) for i in range(num_agents)]
elif algorithm == "sac":
    from tests.rl.algorithms.sac import get_sac_policy, get_sac_trainer

    policies = [
        get_sac_policy(
            f"{algorithm}_{i}.policy",
            action_lower_bound,
            action_upper_bound,
            gym_state_dim,
            gym_action_dim,
            action_limit,
        )
        for i in range(num_agents)
    ]
    trainers = [get_sac_trainer(f"{algorithm}_{i}", gym_state_dim, gym_action_dim) for i in range(num_agents)]
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
