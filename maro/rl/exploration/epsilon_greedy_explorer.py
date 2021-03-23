# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

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
        if isinstance(action_index, np.ndarray):
            return [self._get_exploration_action(act) for act in action_index]
        else:
            return self._get_exploration_action(action_index)

    def set_parameters(self, *, epsilon: float):
        self._epsilon = epsilon

    def _get_exploration_action(self, action_index):
        assert (action_index < self._num_actions), f"Invalid action: {action_index}"
        return action_index if np.random.random() > self._epsilon else np.random.choice(self._num_actions)
