# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch.optim import SGD, Adam

from maro.rl.model import DiscreteACBasedNet, FullyConnected, VNet
from maro.rl.policy import DiscretePolicyGradient
from maro.rl.training.algorithms import ActorCriticParams, ActorCriticTrainer

actor_net_conf = {
    "hidden_dims": [64, 32, 32],
    "activation": torch.nn.LeakyReLU,
    "softmax": True,
    "batch_norm": False,
    "head": True,
}

critic_net_conf = {
    "hidden_dims": [256, 128, 64],
    "activation": torch.nn.LeakyReLU,
    "softmax": False,
    "batch_norm": False,
    "head": True,
}

actor_learning_rate = 0.0001
critic_learning_rate = 0.001


class MyActorNet(DiscreteACBasedNet):
    def __init__(self, state_dim: int, action_num: int, num_features: int) -> None:
        super(MyActorNet, self).__init__(state_dim=state_dim, action_num=action_num)
        self._num_features = num_features
        self._actor = FullyConnected(input_dim=num_features, output_dim=action_num, **actor_net_conf)
        self._optim = Adam(self._actor.parameters(), lr=actor_learning_rate)

    def _get_action_probs_impl(self, states: torch.Tensor) -> torch.Tensor:
        features, masks = states[:, : self._num_features], states[:, self._num_features :]
        masks += 1e-8  # this is to prevent zero probability and infinite logP.
        return self._actor(features) * masks


class MyCriticNet(VNet):
    def __init__(self, state_dim: int, num_features: int) -> None:
        super(MyCriticNet, self).__init__(state_dim=state_dim)
        self._num_features = num_features
        self._critic = FullyConnected(input_dim=num_features, output_dim=1, **critic_net_conf)
        self._optim = SGD(self._critic.parameters(), lr=critic_learning_rate)

    def _get_v_values(self, states: torch.Tensor) -> torch.Tensor:
        features, masks = states[:, : self._num_features], states[:, self._num_features :]
        masks += 1e-8  # this is to prevent zero probability and infinite logP.
        return self._critic(features).squeeze(-1)


def get_ac_policy(state_dim: int, action_num: int, num_features: int, name: str) -> DiscretePolicyGradient:
    return DiscretePolicyGradient(name=name, policy_net=MyActorNet(state_dim, action_num, num_features))


def get_ac(state_dim: int, num_features: int, name: str) -> ActorCriticTrainer:
    return ActorCriticTrainer(
        name=name,
        params=ActorCriticParams(
            get_v_critic_net_func=lambda: MyCriticNet(state_dim, num_features),
            reward_discount=0.9,
            grad_iters=100,
            critic_loss_cls=torch.nn.MSELoss,
            min_logp=-20,
            lam=0.0,
        ),
    )
