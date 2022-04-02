# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict

import torch
from torch.optim import Adam, RMSprop

from maro.rl.model import DiscretePolicyNet, FullyConnected, VNet
from maro.rl.policy import DiscretePolicyGradient
from maro.rl.training.algorithms import DiscretePPOParams, DiscretePPOTrainer, DiscreteActorCriticTrainer, DiscreteActorCriticParams

actor_net_conf = {
    "hidden_dims": [256, 256, 128],
    "activation": torch.nn.Tanh,
    "softmax": True,
    "batch_norm": False,
    "head": True,
}
critic_net_conf = {
    "hidden_dims": [256, 256, 128],
    "output_dim": 1,
    "activation": torch.nn.Tanh,
    "softmax": False,
    "batch_norm": True,
    "head": True,
}
actor_learning_rate = 0.001
critic_learning_rate = 0.001


class MyActorNet(DiscretePolicyNet):
    def __init__(self, state_dim: int, action_num: int) -> None:
        super(MyActorNet, self).__init__(state_dim=state_dim, action_num=action_num)
        self._actor = FullyConnected(input_dim=state_dim, output_dim=action_num, **actor_net_conf)
        self._optim = Adam(self._actor.parameters(), lr=actor_learning_rate)

    def _get_action_probs_impl(self, states: torch.Tensor) -> torch.Tensor:
        return self._actor(states)

    def freeze(self) -> None:
        self.freeze_all_parameters()

    def unfreeze(self) -> None:
        self.unfreeze_all_parameters()

    def step(self, loss: torch.Tensor) -> None:
        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

    def get_gradients(self, loss: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._optim.zero_grad()
        loss.backward()
        return {name: param.grad for name, param in self.named_parameters()}

    def apply_gradients(self, grad: Dict[str, torch.Tensor]) -> None:
        for name, param in self.named_parameters():
            param.grad = grad[name]
        self._optim.step()

    def get_state(self) -> dict:
        return {
            "network": self.state_dict(),
            "optim": self._optim.state_dict(),
        }

    def set_state(self, net_state: dict) -> None:
        self.load_state_dict(net_state["network"])
        self._optim.load_state_dict(net_state["optim"])


class MyCriticNet(VNet):
    def __init__(self, state_dim: int) -> None:
        super(MyCriticNet, self).__init__(state_dim=state_dim)
        self._critic = FullyConnected(input_dim=state_dim, **critic_net_conf)
        self._optim = RMSprop(self._critic.parameters(), lr=critic_learning_rate)

    def _get_v_values(self, states: torch.Tensor) -> torch.Tensor:
        return self._critic(states).squeeze(-1)

    def step(self, loss: torch.Tensor) -> None:
        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

    def get_gradients(self, loss: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._optim.zero_grad()
        loss.backward()
        return {name: param.grad for name, param in self.named_parameters()}

    def apply_gradients(self, grad: Dict[str, torch.Tensor]) -> None:
        for name, param in self.named_parameters():
            param.grad = grad[name]
        self._optim.step()

    def get_state(self) -> dict:
        return {
            "network": self.state_dict(),
            "optim": self._optim.state_dict(),
        }

    def set_state(self, net_state: dict) -> None:
        self.load_state_dict(net_state["network"])
        self._optim.load_state_dict(net_state["optim"])

    def freeze(self) -> None:
        self.freeze_all_parameters()

    def unfreeze(self) -> None:
        self.unfreeze_all_parameters()


def get_policy(state_dim: int, action_num: int, name: str) -> DiscretePolicyGradient:
    policy = DiscretePolicyGradient(name=name, policy_net=MyActorNet(state_dim, action_num))
    return policy


def get_ppo(state_dim: int, name: str) -> DiscreteActorCriticTrainer:
    return DiscreteActorCriticTrainer(
        name=name,
        params=DiscreteActorCriticParams(
            get_v_critic_net_func=lambda: MyCriticNet(state_dim),
            reward_discount=.99,
            grad_iters=20,
            critic_loss_cls=torch.nn.SmoothL1Loss,
            min_logp=-4.0,
            lam=0.99,
            replay_memory_capacity=180
        ),
    )
