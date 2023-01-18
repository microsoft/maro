# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl.training.algorithms import PPOParams, PPOTrainer
from .ac import MyCriticNet, get_ac_policy

get_ppo_policy = get_ac_policy


def get_ppo_trainer(name: str, state_dim: int) -> PPOTrainer:
    return PPOTrainer(
        name=name,
        reward_discount=0.99,
        params=PPOParams(
            get_v_critic_net_func=lambda: MyCriticNet(state_dim),
            grad_iters=80,
            lam=0.97,
            clip_ratio=0.2,
        ),
    )
