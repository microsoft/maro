# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import Optional, Tuple

import torch
from torch.optim import RMSprop

from maro.rl.exploration import EpsilonGreedy
from maro.rl.model import DiscreteQNet, FullyConnected
from maro.rl.policy import ValueBasedPolicy
from maro.rl.training.algorithms import DQNParams, DQNTrainer

q_net_conf = {
    "hidden_dims": [256, 128, 64, 32],
    "activation": torch.nn.LeakyReLU,
    "output_activation": torch.nn.LeakyReLU,
    "softmax": False,
    "batch_norm": True,
    "skip_connection": False,
    "head": True,
    "dropout_p": 0.0,
}
learning_rate = 0.05


class MyQNet(DiscreteQNet):
    def __init__(
        self,
        state_dim: int,
        action_num: int,
        dueling_param: Optional[Tuple[dict, dict]] = None,
    ) -> None:
        super(MyQNet, self).__init__(state_dim=state_dim, action_num=action_num)

        self._use_dueling = dueling_param is not None
        self._fc = FullyConnected(input_dim=state_dim, output_dim=0 if self._use_dueling else action_num, **q_net_conf)
        if self._use_dueling:
            q_kwargs, v_kwargs = dueling_param
            self._q = FullyConnected(input_dim=self._fc.output_dim, output_dim=action_num, **q_kwargs)
            self._v = FullyConnected(input_dim=self._fc.output_dim, output_dim=1, **v_kwargs)

        self._optim = RMSprop(self.parameters(), lr=learning_rate)

    def _get_q_values_for_all_actions(self, states: torch.Tensor) -> torch.Tensor:
        logits = self._fc(states)
        if self._use_dueling:
            q = self._q(logits)
            v = self._v(logits)
            logits = q - q.mean(dim=1, keepdim=True) + v
        return logits


def get_dqn_policy(state_dim: int, action_num: int, name: str) -> ValueBasedPolicy:
    q_kwargs = {
        "hidden_dims": [128],
        "activation": torch.nn.LeakyReLU,
        "output_activation": torch.nn.LeakyReLU,
        "softmax": False,
        "batch_norm": True,
        "skip_connection": False,
        "head": True,
        "dropout_p": 0.0,
    }
    v_kwargs = {
        "hidden_dims": [128],
        "activation": torch.nn.LeakyReLU,
        "output_activation": None,
        "softmax": False,
        "batch_norm": True,
        "skip_connection": False,
        "head": True,
        "dropout_p": 0.0,
    }

    return ValueBasedPolicy(
        name=name,
        q_net=MyQNet(
            state_dim,
            action_num,
            dueling_param=(q_kwargs, v_kwargs),
        ),
        explore_strategy=EpsilonGreedy(epsilon=0.4, num_actions=action_num),
        warmup=100,
    )


def get_dqn(name: str) -> DQNTrainer:
    return DQNTrainer(
        name=name,
        reward_discount=0.0,
        replay_memory_capacity=10000,
        batch_size=32,
        params=DQNParams(
            update_target_every=5,
            num_epochs=10,
            soft_update_coef=0.1,
            double=False,
            alpha=1.0,
            beta=1.0,
        ),
    )
