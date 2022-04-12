from typing import Dict, Tuple

import numpy as np
import torch
from torch.distributions import Normal
from torch.optim import Adam

from examples.gym.rl.algorithms.utils import mlp
from examples.gym.rl.env_helper import action_lower_bound, action_upper_bound, gym_action_dim, gym_state_dim
from maro.rl.model import ContinuousACBasedNet, VNet
from maro.rl.policy import ContinuousRLPolicy
from maro.rl.training.algorithms import PPOParams, PPOTrainer

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


class MyActorNet(ContinuousACBasedNet):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(MyActorNet, self).__init__(state_dim=state_dim, action_dim=action_dim)

        log_std = -0.5 * np.ones(action_dim, dtype=np.float32)
        self._log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self._mu_net = mlp(
            [state_dim] + actor_net_conf["hidden_dims"] + [action_dim],
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


class MyCriticNet(VNet):
    def __init__(self, state_dim: int) -> None:
        super(MyCriticNet, self).__init__(state_dim=state_dim)
        self._critic = mlp(
            [state_dim] + critic_net_conf["hidden_dims"] + [1],
            activation=critic_net_conf["activation"],
        )
        self._optim = Adam(self._critic.parameters(), lr=critic_learning_rate)

    def _get_v_values(self, states: torch.Tensor) -> torch.Tensor:
        return self._critic(states.float()).squeeze(-1)


def get_policy(name: str) -> ContinuousRLPolicy:
    return ContinuousRLPolicy(
        name=name,
        action_range=(action_lower_bound, action_upper_bound),
        policy_net=MyActorNet(gym_state_dim, gym_action_dim),
    )


def get_ppo(name: str) -> PPOTrainer:
    return PPOTrainer(
        name=name,
        params=PPOParams(
            get_v_critic_net_func=lambda: MyCriticNet(gym_state_dim),
            reward_discount=0.99,
            grad_iters=80,
            min_logp=None,
            lam=0.97,
            clip_ratio=0.2,
            is_discrete_action=False,
        ),
    )
