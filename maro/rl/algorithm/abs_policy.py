# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod


class AbsFixedPolicy(ABC):
    """Abstract fixed policy class.

    Args:
        config: Settings for the algorithm.
    """
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def choose_action(self, state):
        """Compute an action based on a state object.

        Args:
            state: State object.

        Returns:
            The action to be taken given the state object. It is usually necessary to use an ``ActionShaper`` to convert
            this to an environment executable action.
        """
        raise NotImplementedError


class AbsTrainablePolicy(ABC):
    """Abstract fixed policy class.

    Args:
        config: Settings for the algorithm.
    """
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def choose_action(self, state):
        """Compute an action based on a state object.

        Args:
            state: State object.

        Returns:
            The action to be taken given the state object. It is usually necessary to use an ``ActionShaper`` to convert
            this to an environment executable action.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, experience_obj):
        """Algorithm-specific training logic.

        The parameters are data to train the underlying model on. Algorithm-specific loss and optimization
        should be reflected here.
        """
        raise NotImplementedError
