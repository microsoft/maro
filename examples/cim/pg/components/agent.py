# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

from maro.rl import AbsAgent


class CIMAgent(AbsAgent):
    def train(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray):
        self._algorithm.train(states, actions, rewards)
