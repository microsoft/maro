# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch

from maro.rl.algorithms.abs_algorithm import AbsAlgorithm
from maro.rl.models.learning_model import SingleTaskLearningModel


class PolicyGradientConfig:
    """Configuration for the Policy Gradient (PG) algorithm.

    Args:
        num_actions (int): Number of possible actions.
        reward_decay (float): Reward decay as defined in standard RL terminology.
    """
    __slots__ = ["num_actions", "reward_decay"]

    def __init__(self, num_actions: int, reward_decay: float):
        self.num_actions = num_actions
        self.reward_decay = reward_decay


class PolicyGradient(AbsAlgorithm):
    """Policy Gradient (PG) algorithm.

    The Policy Gradient algorithm base on the policy gradient theorem, a.k.a. REINFORCE.

    Args:
        core_model (SingleTaskLearningModel): Policy model.
        config: Configuration for the PG algorithm.
    """

    def __init__(self, core_model: SingleTaskLearningModel, config: PolicyGradientConfig):
        super().__init__(core_model, config)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._core_model.to(self._device)

    def choose_action(self, state: np.ndarray, epsilon: float = None):
        state = torch.from_numpy(state).unsqueeze(0).to(self._device)   # (1, state_dim)
        self._core_model.eval()
        with torch.no_grad():
            action_dist = self._core_model(state).squeeze().numpy()  # (num_actions,)
        return np.random.choice(self._config.num_actions, p=action_dist)

    def train(self, states: np.ndarray, actions: np.ndarray, returns: np.ndarray):
        states = torch.from_numpy(states).to(self._device)   # (N, state_dim)
        actions = torch.from_numpy(actions).to(self._device)  # (N,)
        returns = torch.from_numpy(returns).to(self._device)
        action_prob = self._core_model(states).gather(1, actions.unsqueeze(1)).squeeze()   # (N, 1)
        loss = -(torch.log(action_prob) * returns).mean()
        self._core_model.step(loss)
