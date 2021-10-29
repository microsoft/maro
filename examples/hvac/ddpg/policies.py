# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

import numpy as np
import torch

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
    # def __init__(self, out_min=None, out_max=None):
        super().__init__(
            out_min=output_lower_bound,
            out_max=output_upper_bound
        )

        self._input_dim = input_dim
        self._output_dim = output_dim

        self.actor = FullyConnected(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=actor_hidden_dims,
            activation=actor_activation
        )
        self.actor_optimizer = actor_optimizer(
            params=self.actor.parameters(),
            lr=actor_lr
        )

        self.critic = FullyConnected(
            input_dim=input_dim + output_dim,
            output_dim=1,
            hidden_dims=critic_hidden_dims,
            activation=critic_activation
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
            return self.actor(states)
        else:
            return self.critic(torch.cat([states, actions], dim=1))

    def step(self, loss: torch.tensor):
        # TODO: to be confirmed
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
