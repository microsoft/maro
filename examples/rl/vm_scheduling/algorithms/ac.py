# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict

import torch
from torch.optim import Adam, SGD

from maro.rl.model import DiscretePolicyNet, FullyConnected, VNet
from maro.rl.policy import DiscretePolicyGradient
from maro.rl.training.algorithms import DiscreteActorCritic, DiscreteActorCriticParams


actor_net_conf = {
    "hidden_dims": [64, 32, 32],
    "activation": torch.nn.LeakyReLU,
    "softmax": True,
    "batch_norm": False,
    "head": True
}

critic_net_conf = {
    "hidden_dims": [256, 128, 64],
    "activation": torch.nn.LeakyReLU,
    "softmax": False,
    "batch_norm": False,
    "head": True
}

actor_learning_rate = 0.0001
critic_learning_rate = 0.001


class MyActorNet(DiscretePolicyNet):
    def __init__(self, state_dim: int, action_num: int) -> None:
        super(MyActorNet, self).__init__(state_dim=state_dim, action_num=action_num)
        self._actor = FullyConnected(input_dim=state_dim, output_dim=action_num, **actor_net_conf)
        self._actor_optim = Adam(self._actor.parameters(), lr=actor_learning_rate)

    def _get_action_probs_impl(self, states: torch.Tensor) -> torch.Tensor:
        features, masks = states[:, :self._state_dim], states[:, self._state_dim:]
        masks += 1e-8  # this is to prevent zero probability and infinite logP. 
        return self._actor(features) * masks

    def freeze(self) -> None:
        self.freeze_all_parameters()

    def unfreeze(self) -> None:
        self.unfreeze_all_parameters()

    def get_gradients(self, loss: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._actor_optim.zero_grad()
        loss.backward()
        return {name: param.grad for name, param in self.named_parameters()}

    def apply_gradients(self, grad: dict) -> None:
        for name, param in self.named_parameters():
            param.grad = grad[name]
        self._actor_optim.step()

    def get_net_state(self) -> dict:
        return {
            "network": self.state_dict(),
            "actor_optim": self._actor_optim.state_dict()
        }

    def set_net_state(self, net_state: dict) -> None:
        self.load_state_dict(net_state["network"])
        self._actor_optim.load_state_dict(net_state["actor_optim"])


class MyCriticNet(VNet):
    def __init__(self, state_dim: int) -> None:
        super(MyCriticNet, self).__init__(state_dim=state_dim)
        self._critic = FullyConnected(input_dim=state_dim, **critic_net_conf)
        self._critic_optim = SGD(self._critic.parameters(), lr=critic_learning_rate)

    def _get_v_values(self, states: torch.Tensor) -> torch.Tensor:
        features, masks = states[:, :self._state_dim], states[:, self._state_dim:]
        masks += 1e-8  # this is to prevent zero probability and infinite logP. 
        return self._critic(features).squeeze(-1)

    def get_gradients(self, loss: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._critic_optim.zero_grad()
        loss.backward()
        return {name: param.grad for name, param in self.named_parameters()}

    def apply_gradients(self, grad: dict) -> None:
        for name, param in self.named_parameters():
            param.grad = grad[name]
        self._critic_optim.step()

    def get_net_state(self) -> dict:
        return {
            "network": self.state_dict(),
            "critic_optim": self._critic_optim.state_dict()
        }

    def set_net_state(self, net_state: dict) -> None:
        self.load_state_dict(net_state["network"])
        self._critic_optim.load_state_dict(net_state["critic_optim"])

    def freeze(self) -> None:
        self.freeze_all_parameters()

    def unfreeze(self) -> None:
        self.unfreeze_all_parameters()


def get_discrete_policy_gradient(name: str, *, state_dim: int, action_num: int) -> DiscretePolicyGradient:
    return DiscretePolicyGradient(name=name, policy_net=MyActorNet(state_dim, action_num))


def get_ac(name: str, *, state_dim: int) -> DiscreteActorCritic:
    return DiscreteActorCritic(
        name=name,
        params=DiscreteActorCriticParams(
            device="cpu",
            get_v_critic_net_func=lambda: MyCriticNet(state_dim),
            reward_discount=0.9,
            grad_iters=100,
            critic_loss_cls=torch.nn.MSELoss,
            min_logp=-20,
            lam=.0
        )
    )
