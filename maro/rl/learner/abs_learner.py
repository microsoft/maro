# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod


class AbsLearner(ABC):
    def __init__(self):
        """It is used to control the policy learning process...
        """
        pass

    @abstractmethod
    def train(self, total_episodes):
        """The outermost training loop logic is implemented here.

        Args:
            total_episodes (int): number of episodes to be run.
        """
        pass

    @abstractmethod
    def test(self):
        pass
