# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

from maro.rl.rl_component.rl_component_bundle import RLComponentBundle
from maro.rl.training.algorithms.ppo import PPOParams, PPOTrainer

from tests.rl.gym_wrapper.common import (
    action_lower_bound,
    action_upper_bound,
    gym_action_dim,
    gym_state_dim,
    is_discrete,
    learn_env,
    num_agents,
    test_env,
)
from tests.rl.gym_wrapper.env_sampler import GymEnvSampler
from tests.rl.tasks.ac import MyVCriticNet, get_ac_policy

get_ppo_policy = get_ac_policy


def get_ppo_trainer(name: str, state_dim: int) -> PPOTrainer:
    return PPOTrainer(
        name=name,
        reward_discount=0.99,
        replay_memory_capacity=4000,
        batch_size=4000,
        params=PPOParams(
            get_v_critic_net_func=lambda: MyVCriticNet(state_dim),
            grad_iters=80,
            lam=0.97,
            clip_ratio=0.2,
        ),
    )


assert not is_discrete

algorithm = "ppo"
agent2policy = {agent: f"{algorithm}_{agent}.policy" for agent in learn_env.agent_idx_list}
policies = [
    get_ppo_policy(f"{algorithm}_{i}.policy", action_lower_bound, action_upper_bound, gym_state_dim, gym_action_dim)
    for i in range(num_agents)
]
trainers = [get_ppo_trainer(f"{algorithm}_{i}", gym_state_dim) for i in range(num_agents)]

device_mapping = None
if torch.cuda.is_available():
    device_mapping = {f"{algorithm}_{i}.policy": "cuda:0" for i in range(num_agents)}

rl_component_bundle = RLComponentBundle(
    env_sampler=GymEnvSampler(
        learn_env=learn_env,
        test_env=test_env,
        policies=policies,
        agent2policy=agent2policy,
        max_episode_length=1000,
    ),
    agent2policy=agent2policy,
    policies=policies,
    trainers=trainers,
    device_mapping=device_mapping,
)

__all__ = ["rl_component_bundle"]
