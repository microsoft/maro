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
        self.action_space = None

    def set_action_space(self, action_space):
        self.action_space = action_space

    @abstractmethod
    def __call__(self, action, state=None):
        """Generate an exploratory action.

        Args:
            action: Optimal action according to the policy to which this exploration instance belongs.
            state: State information which might be needed as context to generate an exploratory action.
                Defaults to None.
        """
        raise NotImplementedError


class EpsilonGreedyExploration(DiscreteSpaceExploration):
    """Epsilon greedy exploration for discrete action spaces.

    Args:
        num_actions (int): Number of all possible actions.
    """
    def __init__(self, epsilon: float = .0):
        super().__init__()
        self.epsilon = epsilon

    def __call__(self, action: Union[int, np.ndarray], state=None):
        """Generate an exploratory action.

        Args:
            action: Optimal action according to the policy to which this exploration instance belongs.
            state: State information which might be needed as context to generate an exploratory action.
                In this simple epsilon-greedy scheme, it is not used. Defaults to None.
        """
        if isinstance(action, np.ndarray):
            return np.array([self._get_exploration_action(act) for act in action])
        else:
            return self._get_exploration_action(action)

    def _get_exploration_action(self, action):
        return action if np.random.random() > self.epsilon else np.random.choice(self.action_space)
