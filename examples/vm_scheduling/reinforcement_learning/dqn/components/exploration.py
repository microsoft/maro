# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Union

import numpy as np

from maro.rl import EpsilonGreedyExploration


class VMExploration(EpsilonGreedyExploration):
    def __call__(self, action_index: Union[int, np.ndarray], legal_action: np.ndarray):
        if isinstance(action_index, np.ndarray):
            return np.array([self._get_exploration_action(act) for act in action_index])
        else:
            return self._get_exploration_action(action_index, legal_action)

    def _get_exploration_action(self, action_index: int, legal_action: np.ndarray):
        assert (action_index < self._num_actions), f"Invalid action: {action_index}"
        return action_index if np.random.random() > self.epsilon else np.random.choice(np.where(legal_action == 1)[0])
