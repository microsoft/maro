# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random

from .abs_explorer import AbsExplorer


class EpsilonGreedyExplorer(AbsExplorer):
    """Epsilon greedy explorer for discrete action spaces.

    Args:
        num_actions (int): Number of all possible actions.
    """
    def __init__(self, num_actions: int, epsilon: float = .0):
        super().__init__()
        self._num_actions = num_actions
        self._epsilon = epsilon

    def __call__(self, action_index: int):
        assert (action_index < self._num_actions), f"Invalid action: {action_index}"
        if random.random() > self._epsilon:
            return action_index
        else:
            return random.randrange(self._num_actions)

    def update(self, *, epsilon: float):
        self._epsilon = epsilon
