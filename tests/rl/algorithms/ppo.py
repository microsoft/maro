# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Union

import torch

from maro.rl.policy import ContinuousRLPolicy, DiscretePolicyGradient
from maro.rl.training.algorithms import PPOParams, PPOTrainer

from .ac import MyCriticNet, get_ac_policy


def get_ppo_policy(
    name: str,
    state_dim: int,
    action_dim: int,
    is_continuous_action: bool,
    action_lower_bound=None,
    action_upper_bound=None,
) -> Union[ContinuousRLPolicy, DiscretePolicyGradient]:
    return get_ac_policy(name, state_dim, action_dim, is_continuous_action, action_lower_bound, action_upper_bound)

def get_ppo_trainer(name: str, state_dim: int) -> PPOTrainer:
    return PPOTrainer(
        name=name,
        reward_discount=0.99,
        params=PPOParams(
            get_v_critic_net_func=lambda: MyCriticNet(state_dim),
            grad_iters=80,
            critic_loss_cls=torch.nn.SmoothL1Loss,
            lam=0.97,
            clip_ratio=0.2,
        ),
    )
