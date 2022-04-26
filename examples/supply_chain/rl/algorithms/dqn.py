# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict

import numpy as np

import torch
from torch.optim import RMSprop, Adam

from maro.rl.exploration import LinearExplorationScheduler, epsilon_greedy
from examples.supply_chain.rl.exploration import or_epsilon_greedy
from maro.rl.model import DiscreteQNet, FullyConnected
from maro.rl.policy import ValueBasedPolicy
from maro.rl.training.algorithms import DQNParams, DQNTrainer
from maro.rl.utils import match_shape, ndarray_to_tensor


q_net_conf = {
    # "input_dim" will be filled by env_info.py
    "hidden_dims": [256, 256, 256],
    "activation": torch.nn.Tanh,
    "softmax": True,
    "batch_norm": False,
    "skip_connection": False,
    "head": True,
    "dropout_p": 0.1,
}

learning_rate = 0.0005


class MyQNet(DiscreteQNet):
    def __init__(self, state_dim: int, action_num: int) -> None:
        super(MyQNet, self).__init__(state_dim=state_dim, action_num=action_num)
        self._fc = FullyConnected(input_dim=state_dim, output_dim=action_num, **q_net_conf)
        self._optim = Adam(self._fc.parameters(), lr=learning_rate)

    def _get_q_values_for_all_actions(self, states: torch.Tensor) -> torch.Tensor:
        return self._fc(states)

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

import sys

class ORValueBasedPolicy(ValueBasedPolicy):
    def _get_actions_impl(self, states: torch.Tensor, exploring: bool) -> torch.Tensor:
        self._call_cnt += 1
        if self._call_cnt <= self._warmup:
            return ndarray_to_tensor(np.random.randint(self.action_num, size=(states.shape[0], 1)), self._device)

        q_matrix = self.q_values_for_all_actions_tensor(states)  # [B, action_num]
        _, actions = q_matrix.max(dim=1)  # [B], [B]
        if self._exploring:
            or_actions = states[:, -1]
            actions = self._exploration_func(states, actions.cpu().numpy(), self.action_num, or_actions.cpu().numpy(), **self._exploration_params)
            actions = ndarray_to_tensor(actions, self._device)
        return actions.unsqueeze(1)  # [B, 1]

    def explore(self) -> None:
        self._exploring = True
    
    def exploit(self) -> None:
        self._exploring = False


def get_policy(state_dim: int, action_num: int, name: str) -> ValueBasedPolicy:
    policy = ORValueBasedPolicy(
        name=name,
        q_net=MyQNet(state_dim, action_num),
        exploration_strategy=(or_epsilon_greedy, {"epsilon": 2.0}),
        exploration_scheduling_options=[(
            "epsilon", LinearExplorationScheduler, {
            "last_ep": 1000,
            "initial_value": 2.0,
            "final_value": 1.0,
            }
        )],
        warmup=0
    )
    return policy

# def get_policy(state_dim: int, action_num: int, name: str) -> ValueBasedPolicy:
#     policy = ValueBasedPolicy(
#         name=name,
#         q_net=MyQNet(state_dim, action_num),
#         exploration_strategy=(epsilon_greedy, {"epsilon": 1.0}),
#         exploration_scheduling_options=[(
#             "epsilon", LinearExplorationScheduler, {
#             "last_ep": 1000,
#             "initial_value": 1.0,
#             "final_value": 0.0,
#             }
#         )],
#         warmup=1000
#     )
#     return policy


def get_dqn(name: str) -> DQNTrainer:
    return DQNTrainer(
        name=name,
        params=DQNParams(
            reward_discount=.99,
            update_target_every=4,
            num_epochs=100,
            soft_update_coef=0.01,
            double=True,
            replay_memory_capacity=1024000,
            random_overwrite=False,
            batch_size=1024,
        ),
    )
