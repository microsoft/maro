# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import partial
from typing import List

import torch
from torch.optim import Adam, RMSprop

from maro.rl.model import DiscreteACBasedNet, FullyConnected, MultiQNet
from maro.rl.policy import DiscretePolicyGradient
from maro.rl.training.algorithms import DiscreteMADDPGParams, DiscreteMADDPGTrainer

actor_net_conf = {
    "hidden_dims": [256, 128, 64],
    "activation": torch.nn.Tanh,
    "softmax": True,
    "batch_norm": False,
    "head": True,
}
critic_net_conf = {
    "hidden_dims": [256, 128, 64],
    "output_dim": 1,
    "activation": torch.nn.LeakyReLU,
    "softmax": False,
    "batch_norm": True,
    "head": True,
}
actor_learning_rate = 0.001
critic_learning_rate = 0.001


# #####################################################################################################################
class MyActorNet(DiscreteACBasedNet):
    def __init__(self, state_dim: int, action_num: int) -> None:
        super(MyActorNet, self).__init__(state_dim=state_dim, action_num=action_num)
        self._actor = FullyConnected(input_dim=state_dim, output_dim=action_num, **actor_net_conf)
        self._optim = Adam(self._actor.parameters(), lr=actor_learning_rate)

    def _get_action_probs_impl(self, states: torch.Tensor) -> torch.Tensor:
        return self._actor(states)


class MyMultiCriticNet(MultiQNet):
    def __init__(self, state_dim: int, action_dims: List[int]) -> None:
        super(MyMultiCriticNet, self).__init__(state_dim=state_dim, action_dims=action_dims)
        self._critic = FullyConnected(input_dim=state_dim + sum(action_dims), **critic_net_conf)
        self._optim = RMSprop(self._critic.parameters(), critic_learning_rate)

    def _get_q_values(self, states: torch.Tensor, actions: List[torch.Tensor]) -> torch.Tensor:
        return self._critic(torch.cat([states] + actions, dim=1)).squeeze(-1)


def get_multi_critic_net(state_dim: int, action_dims: List[int]) -> MyMultiCriticNet:
    return MyMultiCriticNet(state_dim, action_dims)


def get_maddpg_policy(state_dim: int, action_num: int, name: str) -> DiscretePolicyGradient:
    return DiscretePolicyGradient(name=name, policy_net=MyActorNet(state_dim, action_num))


def get_maddpg(state_dim: int, action_dims: List[int], name: str) -> DiscreteMADDPGTrainer:
    return DiscreteMADDPGTrainer(
        name=name,
        params=DiscreteMADDPGParams(
            reward_discount=0.0,
            num_epoch=10,
            get_q_critic_net_func=partial(get_multi_critic_net, state_dim, action_dims),
            shared_critic=False,
        ),
    )
