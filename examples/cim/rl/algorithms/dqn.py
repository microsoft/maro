# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict

import torch
from torch.optim import RMSprop

from maro.rl.exploration import MultiLinearExplorationScheduler, epsilon_greedy
from maro.rl.model import DiscreteQNet, FullyConnected
from maro.rl.policy import ValueBasedPolicy
from maro.rl.training.algorithms import DQNTrainer, DQNParams

q_net_conf = {
    "hidden_dims": [256, 128, 64, 32],
    "activation": torch.nn.LeakyReLU,
    "softmax": False,
    "batch_norm": True,
    "skip_connection": False,
    "head": True,
    "dropout_p": 0.0
}
learning_rate = 0.05


class MyQNet(DiscreteQNet):
    def __init__(self, state_dim: int, action_num: int) -> None:
        super(MyQNet, self).__init__(state_dim=state_dim, action_num=action_num)
        self._fc = FullyConnected(input_dim=state_dim, output_dim=action_num, **q_net_conf)
        self._optim = RMSprop(self._fc.parameters(), lr=learning_rate)

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


def get_policy(state_dim: int, action_num: int, name: str) -> ValueBasedPolicy:
    return ValueBasedPolicy(
        name=name,
        q_net=MyQNet(state_dim, action_num),
        exploration_strategy=(epsilon_greedy, {"epsilon": 0.4}),
        exploration_scheduling_options=[(
            "epsilon", MultiLinearExplorationScheduler, {
                "splits": [(2, 0.32)],
                "initial_value": 0.4,
                "last_ep": 5,
                "final_value": 0.0,
            }
        )],
        warmup=100
    )


def get_dqn(name: str) -> DQNTrainer:
    return DQNTrainer(
        name=name,
        params=DQNParams(
            reward_discount=.0,
            update_target_every=5,
            num_epochs=10,
            soft_update_coef=0.1,
            double=False,
            replay_memory_capacity=10000,
            random_overwrite=False,
            batch_size=32
        ),
    )
