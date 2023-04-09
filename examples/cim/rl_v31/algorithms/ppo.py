# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import random

import numpy as np
import torch
from gym import spaces
from torch.optim import Adam, RMSprop

from maro.rl.model import FullyConnected
from maro.rl_v31.model.base import BaseNet, PolicyModel
from maro.rl_v31.model.vnet import VCritic
from maro.rl_v31.policy import PPOPolicy

actor_net_conf = {
    "hidden_dims": [256, 128, 64],
    "activation": torch.nn.Tanh,
    "output_activation": torch.nn.Tanh,
    "softmax": True,
    "batch_norm": False,
    "head": True,
}
critic_net_conf = {
    "hidden_dims": [256, 128, 64],
    "output_dim": 1,
    "activation": torch.nn.LeakyReLU,
    "output_activation": torch.nn.LeakyReLU,
    "softmax": False,
    "batch_norm": True,
    "head": True,
}
actor_learning_rate = 0.001
critic_learning_rate = 0.001


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class MyPolicyModel(PolicyModel):
    def __init__(self, state_dim: int, action_num: int) -> None:
        super().__init__()

        self.mlp = FullyConnected(input_dim=state_dim, output_dim=action_num, **actor_net_conf)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


class MyCriticModel(BaseNet):
    def __init__(self, state_dim: int) -> None:
        super().__init__()

        self.mlp = FullyConnected(input_dim=state_dim, **critic_net_conf)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs).squeeze(-1)


def get_ppo_policy(state_dim: int, action_num: int, name: str) -> PPOPolicy:
    obs_space = spaces.Box(-np.inf, np.inf, shape=(state_dim,))
    action_space = spaces.Discrete(action_num)
    model = MyPolicyModel(state_dim=state_dim, action_num=action_num)
    optim = Adam(model.parameters(), lr=actor_learning_rate)

    return PPOPolicy(
        name=name,
        obs_space=obs_space,
        action_space=action_space,
        model=model,
        optim=optim,
        is_discrete=True,
    )


def get_ppo_critic(state_dim: int) -> VCritic:
    model = MyCriticModel(state_dim=state_dim)
    optim = RMSprop(model.parameters(), lr=critic_learning_rate)

    return VCritic(model=model, optim=optim)
