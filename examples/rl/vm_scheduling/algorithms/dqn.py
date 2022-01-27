# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict

import numpy as np
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from maro.rl.exploration import MultiLinearExplorationScheduler
from maro.rl.model import DiscreteQNet, FullyConnected
from maro.rl.policy import ValueBasedPolicy
from maro.rl.training.algorithms import DQN, DQNParams


q_net_conf = {
    "hidden_dims": [64, 128, 256],
    "activation": torch.nn.LeakyReLU,
    "softmax": False,
    "batch_norm": False,
    "skip_connection": False,
    "head": True,
    "dropout_p": 0.0
}
q_net_learning_rate = 0.0005
q_net_lr_scheduler_params = {"T_0": 500, "T_mult": 2}


class MyQNet(DiscreteQNet):
    def __init__(self, state_dim: int, action_num: int, num_features: int) -> None:
        super(MyQNet, self).__init__(state_dim=state_dim, action_num=action_num)
        self._num_features = num_features
        self._fc = FullyConnected(input_dim=num_features, output_dim=action_num, **q_net_conf)
        self._optim = SGD(self._fc.parameters(), lr=q_net_learning_rate)
        self._lr_scheduler = CosineAnnealingWarmRestarts(self._optim, **q_net_lr_scheduler_params)

    def _get_q_values_for_all_actions(self, states: torch.Tensor) -> torch.Tensor:
        masks = states[:, self._num_features:]
        q_for_all_actions = self._fc(states[:, :self._num_features])
        return q_for_all_actions + (masks - 1) * 1e8

    def step(self, loss: torch.Tensor) -> None:
        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

    def get_gradients(self, loss: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._optim.zero_grad()
        loss.backward()
        return {name: param.grad for name, param in self.named_parameters()}

    def apply_gradients(self, grad: dict) -> None:
        for name, param in self.named_parameters():
            param.grad = grad[name]
        self._optim.step()

    def get_state(self) -> object:
        return {"network": self.state_dict(), "optim": self._optim.state_dict()}

    def set_state(self, net_state: object) -> None:
        assert isinstance(net_state, dict)
        self.load_state_dict(net_state["network"])
        self._optim.load_state_dict(net_state["optim"])

    def freeze(self) -> None:
        self.freeze_all_parameters()

    def unfreeze(self) -> None:
        self.unfreeze_all_parameters()


class MaskedEpsGreedy:
    def __init__(self, state_dim: int, num_features: int) -> None:
        self._state_dim = state_dim
        self._num_features = num_features

    def __call__(self, states, actions, num_actions, *, epsilon):
        masks = states[:, self._num_features:]
        return np.array([
            action if np.random.random() > epsilon else np.random.choice(np.where(mask == 1)[0])
            for action, mask in zip(actions, masks)
        ])


def get_policy(state_dim: int, action_num: int, num_features: int, name: str) -> ValueBasedPolicy:
    return ValueBasedPolicy(
        name=name,
        q_net=MyQNet(state_dim, action_num, num_features),
        exploration_strategy=(MaskedEpsGreedy(state_dim, num_features), {"epsilon": 0.4}),
        exploration_scheduling_options=[(
            "epsilon", MultiLinearExplorationScheduler, {
                "splits": [(100, 0.32)],
                "initial_value": 0.4,
                "last_ep": 400,
                "final_value": 0.0,
            }
        )],
        warmup=100
    )


def get_dqn(name: str) -> DQN:
    return DQN(
        name=name,
        params=DQNParams(
            device="cpu",
            reward_discount=0.9,
            update_target_every=5,
            num_epochs=100,
            soft_update_coef=0.1,
            double=False,
            replay_memory_capacity=10000,
            random_overwrite=False,
            batch_size=32,
            data_parallelism=2
        )
    )
