# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
from typing import Union

import numpy as np

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

    def __call__(self, action_index: Union[int, np.ndarray]):
        def get_exploration_action(action_index, num_actions, epsilon):
            assert (action_index < num_actions), f"Invalid action: {action_index}"
            return action_index if np.random.random() > epsilon else np.random.choice(num_actions)
        if isinstance(action_index, int):
            return get_exploration_action(action_index, self._num_actions, self._epsilon)
        else:
            return [get_exploration_action(act, self._num_actions, self._epsilon) for act in action_index]

    def update(self, *, epsilon: float):
        self._epsilon = epsilon
