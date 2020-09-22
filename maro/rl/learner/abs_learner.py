# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod


class AbsLearner(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def train(self, total_episodes):
        """
        Main loop for collecting experiences and performance from the actor and using them to optimize models
        Args:
            total_episodes (int): number of episodes for the main training loop
        """
        pass

    @abstractmethod
    def test(self):
        pass
