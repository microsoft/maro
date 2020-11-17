# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random

from .abs_explorer import AbsExplorer


class EpsilonGreedyExplorer(AbsExplorer):
    """Epsilon greedy explorer for discrete action spaces.

    Args:
        num_actions (int): Number of all possible actions.
    """
    def __init__(self, num_actions: int):
        super().__init__()
        self._epsilon_schedule = None
        self._epsilon = None
        self._num_actions = num_actions

    def __call__(self, action):
        assert action < self._num_actions, f"Invalid action: {action}"
        if self._epsilon_schedule is None or random.random() > self._epsilon:
            return action
        else:
            return random.randrange(self._num_actions)

    def load_exploration_params(self, epsilon: float):
        self._epsilon = epsilon
