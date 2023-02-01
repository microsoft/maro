# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Independent, Normal
from torch.optim import Adam

from maro.rl.model import ContinuousSACNet, QNet
from maro.rl.model.fc_block import FullyConnected
from maro.rl.policy import ContinuousRLPolicy
from maro.rl.training.algorithms import SoftActorCriticParams, SoftActorCriticTrainer

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


class MyContinuousSACNet(ContinuousSACNet):
    def __init__(self, state_dim: int, action_dim: int, action_limit: float) -> None:
        super(MyContinuousSACNet, self).__init__(state_dim=state_dim, action_dim=action_dim)

        self._net = FullyConnected(
            input_dim=state_dim,
            output_dim=actor_net_conf["hidden_dims"][-1],
            hidden_dims=actor_net_conf["hidden_dims"][:-1],
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

        pi_distribution = Normal(mu, std)
        pi_action = pi_distribution.rsample() if exploring else mu

        logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)

        pi_action = torch.tanh(pi_action) * self._action_limit

        return pi_action, logp_pi


class MyQCriticNet(QNet):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(MyQCriticNet, self).__init__(state_dim=state_dim, action_dim=action_dim)
        self._critic = FullyConnected(
            input_dim=state_dim + action_dim,
            output_dim=1,
            hidden_dims=critic_net_conf["hidden_dims"],
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
        replay_memory_capacity=200000,
        params=SoftActorCriticParams(
            get_q_critic_net_func=lambda: MyQCriticNet(state_dim, action_dim),
            num_epochs=10,
            n_start_train=10000,
            soft_update_coef=0.01,
        ),
    )
