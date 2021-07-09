# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from typing import Union

import numpy as np

from .abs_exploration import AbsExploration


class DiscreteSpaceExploration(AbsExploration):
    """Exploration for discrete action spaces."""
    def __init__(self):
        super().__init__()
        self._action_space = None

    def set_action_space(self, action_space):
        self._action_space = action_space

    @abstractmethod
    def __call__(self, action_index):
        raise NotImplementedError


class EpsilonGreedyExploration(DiscreteSpaceExploration):
    """Epsilon greedy exploration for discrete action spaces.

    Args:
        num_actions (int): Number of all possible actions.
    """
    def __init__(self, epsilon: float = .0):
        super().__init__()
        self.epsilon = epsilon

    def __call__(self, action: Union[int, np.ndarray]):
        if isinstance(action, np.ndarray):
            return np.array([self._get_exploration_action(act) for act in action])
        else:
            return self._get_exploration_action(action)

    def _get_exploration_action(self, action):
        return action if np.random.random() > self.epsilon else np.random.choice(self._action_space)
