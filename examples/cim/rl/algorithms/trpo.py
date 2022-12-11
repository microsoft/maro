# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

from maro.rl.policy import DiscretePolicyGradient
from maro.rl.training.algorithms import TRPOParams, TRPOTrainer

from .ac import MyActorNet, MyCriticNet

def get_trpo_policy(state_dim: int, action_num: int, name: str) -> DiscretePolicyGradient:
    return DiscretePolicyGradient(name=name, policy_net=MyActorNet(state_dim, action_num))

def get_trpo(state_dim: int, name: str) -> TRPOTrainer:
    return TRPOTrainer(
        name=name,
        reward_discount=0.0,
        params=TRPOParams(
            get_v_critic_net_func=lambda: MyCriticNet(state_dim),
            grad_iters=20,
            critic_loss_cls=torch.nn.SmoothL1Loss,
            lam=0.97,
        ),
    )
