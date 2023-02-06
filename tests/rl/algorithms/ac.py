# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Tuple

import numpy as np
import torch
from torch.distributions import Normal
from torch.optim import Adam

from maro.rl.model import ContinuousACBasedNet, VNet
from maro.rl.model.fc_block import FullyConnected
from maro.rl.policy import ContinuousRLPolicy
from maro.rl.training.algorithms import ActorCriticParams, ActorCriticTrainer

actor_net_conf = {
    "hidden_dims": [64, 64],
    "activation": torch.nn.Tanh,
}
critic_net_conf = {
    "hidden_dims": [64, 64],
    "activation": torch.nn.Tanh,
}
actor_learning_rate = 3e-4
critic_learning_rate = 1e-3


class MyContinuousACBasedNet(ContinuousACBasedNet):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(MyContinuousACBasedNet, self).__init__(state_dim=state_dim, action_dim=action_dim)

        log_std = -0.5 * np.ones(action_dim, dtype=np.float32)
        self._log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self._mu_net = FullyConnected(
            input_dim=state_dim,
            hidden_dims=actor_net_conf["hidden_dims"],
            output_dim=action_dim,
            activation=actor_net_conf["activation"],
        )
        self._optim = Adam(self.parameters(), lr=actor_learning_rate)

    def _get_actions_with_logps_impl(self, states: torch.Tensor, exploring: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        distribution = self._distribution(states)
        actions = distribution.sample()
        logps = distribution.log_prob(actions).sum(axis=-1)
        return actions, logps

    def _get_states_actions_logps_impl(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        distribution = self._distribution(states)
        logps = distribution.log_prob(actions).sum(axis=-1)
        return logps

    def _distribution(self, states: torch.Tensor) -> Normal:
        mu = self._mu_net(states.float())
        std = torch.exp(self._log_std)
        return Normal(mu, std)


class MyVCriticNet(VNet):
    def __init__(self, state_dim: int) -> None:
        super(MyVCriticNet, self).__init__(state_dim=state_dim)
        self._critic = FullyConnected(
            input_dim=state_dim,
            output_dim=1,
            hidden_dims=critic_net_conf["hidden_dims"],
            activation=critic_net_conf["activation"],
        )
        self._optim = Adam(self._critic.parameters(), lr=critic_learning_rate)

    def _get_v_values(self, states: torch.Tensor) -> torch.Tensor:
        return self._critic(states.float()).squeeze(-1)


def get_ac_policy(
    name: str,
    action_lower_bound: list,
    action_upper_bound: list,
    gym_state_dim: int,
    gym_action_dim: int,
) -> ContinuousRLPolicy:
    return ContinuousRLPolicy(
        name=name,
        action_range=(action_lower_bound, action_upper_bound),
        policy_net=MyContinuousACBasedNet(gym_state_dim, gym_action_dim),
    )


def get_ac_trainer(name: str, state_dim: int) -> ActorCriticTrainer:
    return ActorCriticTrainer(
        name=name,
        reward_discount=0.99,
        params=ActorCriticParams(
            get_v_critic_net_func=lambda: MyVCriticNet(state_dim),
            grad_iters=80,
            lam=0.97,
        ),
    )
