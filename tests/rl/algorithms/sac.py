# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Independent, Normal
from torch.optim import Adam

from maro.rl.model import ContinuousSACNet, QNet
from maro.rl.policy import ContinuousRLPolicy
from maro.rl.training.algorithms import SoftActorCriticParams, SoftActorCriticTrainer
from tests.rl.algorithms.utils import mlp

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

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class MyContinuousSACNetOld(ContinuousSACNet):
    def __init__(self, state_dim: int, action_dim: int, action_limit: float) -> None:
        super(MyContinuousSACNetOld, self).__init__(state_dim=state_dim, action_dim=action_dim)

        self._net = mlp(
            [state_dim] + actor_net_conf["hidden_dims"],
            activation=actor_net_conf["activation"],
            output_activation=actor_net_conf["activation"],
        )
        self._mu = torch.nn.Linear(actor_net_conf["hidden_dims"][-1], action_dim)
        self._log_std = torch.nn.Linear(actor_net_conf["hidden_dims"][-1], action_dim)
        self._action_limit = action_limit
        self._optim = Adam(self.parameters(), lr=actor_learning_rate)

    def _get_actions_with_logps_impl(self, states: torch.Tensor, exploring: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        net_out = self._net(states.float())
        mu = self._mu(net_out)
        log_std = torch.clamp(self._log_std(net_out), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Independent(Normal(mu, std), 1)
        pi_action = pi_distribution.rsample() if exploring else mu

        logp_pi = pi_distribution.log_prob(pi_action)
        logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)

        return torch.tanh(pi_action) * self._action_limit, logp_pi

class MyContinuousSACNet(ContinuousSACNet):
    def __init__(self, state_dim: int, action_dim: int, action_limit: float) -> None:
        super(MyContinuousSACNet, self).__init__(state_dim=state_dim, action_dim=action_dim)

        log_std = -0.5 * np.ones(action_dim, dtype=np.float32)
        self._log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self._mu_net = mlp(
            [state_dim] + actor_net_conf["hidden_dims"] + [action_dim],
            activation=actor_net_conf["activation"],
        )
        self._optim = Adam(self.parameters(), lr=actor_learning_rate)

        self._action_limit = action_limit

    def _get_actions_with_logps_impl(self, states: torch.Tensor, exploring: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self._mu_net(states.float())
        std = torch.exp(torch.clamp(self._log_std, LOG_STD_MIN, LOG_STD_MAX))  # 1
        distribution = Normal(mu, std)

        actions = distribution.rsample() if exploring else mu
        logps = distribution.log_prob(actions).sum(axis=-1)

        logps -= (2 * (np.log(2) - actions - F.softplus(-2 * actions))).sum(axis=1)  # 2
        actions = torch.tanh(actions) * self._action_limit  # 3

        return actions, logps


class MyQCriticNet(QNet):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(MyQCriticNet, self).__init__(state_dim=state_dim, action_dim=action_dim)
        self._critic = mlp(
            [state_dim + action_dim] + critic_net_conf["hidden_dims"] + [1],
            activation=critic_net_conf["activation"],
        )
        self._optim = Adam(self._critic.parameters(), lr=critic_learning_rate)

    def _get_q_values(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self._critic(torch.cat([states, actions], dim=1).float()).squeeze(-1)


def get_sac_policy(
    name: str,
    action_lower_bound: list,
    action_upper_bound: list,
    gym_state_dim: int,
    gym_action_dim: int,
    action_limit: float,
) -> ContinuousRLPolicy:
    return ContinuousRLPolicy(
        name=name,
        action_range=(action_lower_bound, action_upper_bound),
        policy_net=MyContinuousSACNet(gym_state_dim, gym_action_dim, action_limit),
    )


def get_sac_trainer(name: str, state_dim: int, action_dim: int) -> SoftActorCriticTrainer:
    return SoftActorCriticTrainer(
        name=name,
        reward_discount=0.99,
        params=SoftActorCriticParams(
            get_q_critic_net_func=lambda: MyQCriticNet(state_dim, action_dim),
            num_epochs=50,
            n_start_train=10000,
            soft_update_coef=0.01,
        ),
    )
