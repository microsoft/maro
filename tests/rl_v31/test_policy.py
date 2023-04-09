# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import torch
from gym import spaces
from tianshou.data import Batch
from torch import nn
from torch.optim import Adam

from maro.rl_v31.model.base import PolicyModel
from maro.rl_v31.policy import PPOPolicy


class MyPolicyModel(PolicyModel):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()

        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.fc(obs)


def test_policies() -> None:
    input_dim = 8
    output_dim = 4

    model = MyPolicyModel(input_dim=input_dim, output_dim=output_dim)
    optim = Adam(model.parameters())
    policy = PPOPolicy(
        name="test_ppo",
        obs_space=spaces.Box(-10.0, 10.0, shape=(input_dim,)),
        action_space=spaces.Box(-np.inf, np.inf, shape=(output_dim,)),
        model=MyPolicyModel(input_dim=input_dim, output_dim=output_dim),
        optim=optim,
    )

    print(policy(Batch(obs=np.ones((10, input_dim), dtype=np.float32))))


test_policies()
