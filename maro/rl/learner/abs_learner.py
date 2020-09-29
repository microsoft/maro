# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod


class AbsLearner(ABC):
    """Abstract learner class to control the policy learning process."""
    def __init__(self):
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
        """Test policy performance."""
        pass
