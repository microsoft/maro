# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Categorical

from maro.rl import DiscreteACNet, OptimOption


class ACNet(DiscreteACNet):
    def forward(self, states, actor: bool = True, critic: bool = True):
        states = torch.from_numpy(np.asarray(states)).to(self.device)
        if len(states.shape) == 1:
            states = states.unsqueeze(dim=0)
        return (
            self.component["actor"](states) if actor else None,
            self.component["critic"](states) if critic else None
        )

    def get_action(self, states, legal_action, training=True):
        """
        Given Q-values for a batch of states, return the action index and the corresponding maximum Q-value
        for each state.
        """
        legal_action = torch.from_numpy(np.asarray(legal_action)).to(self.device)

        if not training:
            action_prob = self.forward(states, critic=False)[0]
            _, action = (action_prob + (legal_action - 1) * 1e8).max(dim=1)
            return action, action_prob

        action_prob = Categorical(self.forward(states, critic=False)[0] * legal_action)  # (batch_size, action_space_size)
        action = action_prob.sample()
        log_p = action_prob.log_prob(action)
        return action, log_p
