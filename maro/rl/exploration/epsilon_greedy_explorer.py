# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random

from maro.utils.exception.rl_toolkit_exception import MissingExplorationParametersError

from .abs_explorer import AbsExplorer


class EpsilonGreedyExplorer(AbsExplorer):
    """Epsilon greedy explorer for discrete action spaces.

    Args:
        num_actions (int): Number of all possible actions.
    """
    def __init__(self, num_actions: int):
        super().__init__()
        self._num_actions = num_actions
        self._epsilon = None

    def __call__(self, action):
        assert action < self._num_actions, f"Invalid action: {action}"
        if self._epsilon is None:
            raise MissingExplorationParametersError(
                'Epsilon is not set. Use load_exploration_params with keyword argument "epsilon" to '
                'load the exploration parameters first.'
            )
        if random.random() > self._epsilon:
            return action
        else:
            return random.randrange(self._num_actions)

    def load_exploration_params(self, *, epsilon: float):
        self._epsilon = epsilon
