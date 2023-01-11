# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Tuple, Union
import numpy as np
import torch
from torch.distributions import Normal
from torch.optim import Adam

from maro.rl.model import ContinuousACBasedNet, DiscreteACBasedNet, FullyConnected, VNet
from maro.rl.policy import ContinuousRLPolicy, DiscretePolicyGradient
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


class MyContinuousActorNet(ContinuousACBasedNet):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(MyContinuousActorNet, self).__init__(state_dim=state_dim, action_dim=action_dim)
        self._mu = FullyConnected(input_dim=state_dim, output_dim=action_dim, **actor_net_conf)
        self._std = torch.nn.Parameter(torch.as_tensor(-0.5 * np.ones(action_dim, dtype=np.float32)))
        self._optim = Adam(self.parameters(), lr=actor_learning_rate)

    def _distribution(self, states: torch.Tensor) -> Normal:
        mu = self._mu(states.float())
        std = torch.exp(self._std)
        return Normal(mu, std)

    def _get_actions_with_logps_impl(self, states: torch.Tensor, exploring: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        distribution = self._distribution(states)
        actions = distribution.sample()
        logps = distribution.log_prob(actions).sum(axis=-1)
        return actions, logps

    def _get_states_actions_logps_impl(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        distribution = self._distribution(states)
        logps = distribution.log_prob(actions).sum(axis=-1)
        return logps


class MyDiscreteActorNet(DiscreteACBasedNet):
    def __init__(self, state_dim: int, action_num: int) -> None:
        super(MyDiscreteActorNet, self).__init__(state_dim=state_dim, action_num=action_num)
        self._actor = FullyConnected(input_dim=state_dim, output_dim=action_num, **actor_net_conf)
        self._optim = Adam(self._actor.parameters(), lr=actor_learning_rate)

    def _get_action_probs_impl(self, states: torch.Tensor) -> torch.Tensor:
        return self._actor(states)


class MyCriticNet(VNet):
    def __init__(self, state_dim: int) -> None:
        super(MyCriticNet, self).__init__(state_dim=state_dim)
        self._critic = FullyConnected(input_dim=state_dim, output_dim=1, **critic_net_conf)
        self._optim = Adam(self._critic.parameters(), lr=critic_learning_rate)

    def _get_v_values(self, states: torch.Tensor) -> torch.Tensor:
        return self._critic(states.float()).squeeze(-1)


def get_ac_policy(
    name: str,
    state_dim: int,
    action_dim: int,
    is_continuous_action: bool,
    action_lower_bound=None,
    action_upper_bound=None,
) -> Union[ContinuousRLPolicy, DiscretePolicyGradient]:
    if is_continuous_action:
        assert (action_lower_bound is not None) and (action_upper_bound is not None)
        return ContinuousRLPolicy(
            name=name,
            action_range=(action_lower_bound, action_upper_bound),
            policy_net=MyContinuousActorNet(state_dim, action_dim)
        )
    else:
        return DiscretePolicyGradient(name=name, policy_net=MyDiscreteActorNet(state_dim, action_dim))

def get_ac_trainer(name: str, state_dim: int) -> ActorCriticTrainer:
    return ActorCriticTrainer(
        name=name,
        reward_discount=0.99,
        params=ActorCriticParams(
            get_v_critic_net_func=lambda: MyCriticNet(state_dim),
            grad_iters=80,
            critic_loss_cls=torch.nn.SmoothL1Loss,
            min_logp=None,
            lam=0.97,
        ),
    )
