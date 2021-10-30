# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Union

import numpy as np
import torch
import torch.nn as nn

from maro.rl.modeling import ContinuousACNet, FullyConnected

class AhuACNet(ContinuousACNet):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        output_lower_bound: List[float],
        output_upper_bound: List[float],
        actor_hidden_dims: List[int],
        critic_hidden_dims: List[int],
        actor_activation,
        critic_activation,
        actor_optimizer,
        critic_optimizer,
        actor_lr: float,
        critic_lr: float
    ):
        super().__init__(
            out_min=output_lower_bound,
            out_max=output_upper_bound
        )

        self._input_dim = input_dim
        self._output_dim = output_dim

        self._base_value = torch.tensor(output_lower_bound)
        self._gap_size = torch.tensor(output_upper_bound) - torch.tensor(output_lower_bound)

        self.actor = nn.Sequential(
            FullyConnected(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=actor_hidden_dims,
                activation=actor_activation,
                head=True
            ),
            nn.Tanh()
        )
        self.actor_optimizer = actor_optimizer(
            params=self.actor.parameters(),
            lr=actor_lr
        )

        self.critic = FullyConnected(
            input_dim=input_dim + output_dim,
            output_dim=1,
            hidden_dims=critic_hidden_dims,
            activation=critic_activation,
            head=True
        )
        self.critic_optimizer = critic_optimizer(
            params=self.critic.parameters(),
            lr=critic_lr
        )

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def action_dim(self):
        return self._output_dim

    def forward(self, states: torch.tensor, actions: torch.tensor=None) -> torch.tensor:
        if actions is None:
            return (self.actor(states) + 1) / 2 * self._gap_size + self._base_value
        else:
            return self.critic(torch.cat([states, actions], dim=1))

    def step(self, loss: torch.tensor):
        # TODO: to be confirmed
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def set_state(self, state):
        self.load_state_dict(state["network"]),
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])

    def get_state(self):
        return {
            "network": self.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict()
        }


def relative_gaussian_noise(
    state: np.ndarray,
    action: np.ndarray,
    min_action: Union[float, list, np.ndarray],
    max_action: Union[float, list, np.ndarray],
    mean: Union[float, list, np.ndarray] = .0,
    stddev: Union[float, list, np.ndarray] = 1.0,
) -> Union[float, np.ndarray]:
    noise = np.random.normal(loc=mean, scale=stddev, size=action.shape)
    action = action + noise * (np.array(max_action) - np.array(min_action))
    return np.clip(action, min_action, max_action)
