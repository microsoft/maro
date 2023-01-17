from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam

from maro.rl.model import ContinuousQNet, ContinuousSACNet
from maro.rl.policy import ContinuousRLPolicy
from maro.rl.training.algorithms import SoftActorCriticParams, SoftActorCriticTrainer

from examples.gym.rl.algorithms.utils import mlp
from examples.gym.rl.env_helper import action_lower_bound, action_upper_bound, gym_action_dim, gym_env, gym_state_dim

actor_net_conf = {
    "hidden_dims": [256, 256],
    "activation": torch.nn.ReLU,
}
critic_net_conf = {
    "hidden_dims": [256, 256],
    "activation": torch.nn.ReLU,
}
actor_learning_rate = 1e-3
critic_learning_rate = 1e-3


class MyActorNet(ContinuousSACNet):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(MyActorNet, self).__init__(state_dim=state_dim, action_dim=action_dim)

        self._fc = mlp(
            [state_dim] + actor_net_conf["hidden_dims"],
            activation=actor_net_conf["activation"],
            output_activation=actor_net_conf["activation"],
        )
        self._mu_layer = torch.nn.Linear(actor_net_conf["hidden_dims"][-1], action_dim)
        self._log_std_layer = torch.nn.Linear(actor_net_conf["hidden_dims"][-1], action_dim)
        self._limit = gym_env.action_space.high[0]

        self._optim = Adam(self.parameters(), lr=actor_learning_rate)

    def _get_actions_with_logps_impl(self, states: torch.Tensor, exploring: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        distribution = self._distribution(states)
        if exploring:
            actions = distribution.rsample()
            logps = distribution.log_prob(actions).sum(axis=-1)
            logps -= (2 * (np.log(2) - actions - F.softplus(-2 * actions))).sum(axis=1)
        else:
            actions = distribution.loc
            logps = torch.randn(states.shape[0])  # Fake
        return self._limit * torch.tanh(actions), logps

    def _distribution(self, states: torch.Tensor) -> Normal:
        net_out = self._fc(states.float())
        mu = self._mu_layer(net_out)
        log_std = self._log_std_layer(net_out)
        log_std = torch.clamp(log_std, -20, 2)  # TODO
        std = torch.exp(log_std)
        return Normal(mu, std)


class MyCriticNet(ContinuousQNet):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(MyCriticNet, self).__init__(state_dim=state_dim, action_dim=action_dim)
        self._critic = mlp(
            [state_dim + action_dim] + critic_net_conf["hidden_dims"] + [1],
            activation=critic_net_conf["activation"],
        )
        self._optim = Adam(self._critic.parameters(), lr=critic_learning_rate)

    def _get_q_values(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self._critic(torch.cat([states.float(), actions], dim=1)).reshape(-1)


def get_policy(name: str) -> ContinuousRLPolicy:
    return ContinuousRLPolicy(
        name=name,
        action_range=(action_lower_bound, action_upper_bound),
        policy_net=MyActorNet(gym_state_dim, gym_action_dim),
    )


def get_sac(name: str) -> SoftActorCriticTrainer:
    return SoftActorCriticTrainer(
        name=name,
        params=SoftActorCriticParams(
            replay_memory_capacity=int(1e6),
            batch_size=128,
            get_q_critic_net_func=lambda: MyCriticNet(gym_state_dim, gym_action_dim),
            reward_discount=0.99,
            soft_update_coef=0.01,
            update_target_every=1,
            entropy_coef=0.2,
            num_epochs=100,
            n_start_train=10000,
        ),
    )
