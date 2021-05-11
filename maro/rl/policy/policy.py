# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod

from maro.rl.exploration import AbsExploration
from maro.rl.experience import AbsExperienceManager, ExperienceSet


class AbsPolicy(ABC):
    """Abstract fixed policy class.

    Args:
        config: Settings for the algorithm.
    """
    def __init__(self):
        super().__init__()

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


class NullPolicy(AbsPolicy):
    def choose_action(self, state):
        return None


class AbsCorePolicy(AbsPolicy):
    def __init__(self, experience_manager: AbsExperienceManager, config):
        super().__init__()
        self.experience_manager = experience_manager
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
    def update(self):
        raise NotImplementedError

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def set_state(self, policy_state):
        pass

    def load(self, path: str):
        pass

    def save(self, path: str):
        pass


class RLPolicy(object):
    """Abstract fixed policy class.

    Args:
        config: Settings for the algorithm.
    """
    def __init__(self, core_policy: AbsCorePolicy, exploration: AbsExploration = None):
        self.core_policy = core_policy
        self.exploration = exploration

    def choose_action(self, state):
        """Compute an action based on a state object.

        Args:
            state: State object.

        Returns:
            The action to be taken given the state object. It is usually necessary to use an ``ActionShaper`` to convert
            this to an environment executable action.
        """
        action = self.core_policy.choose_action(state)
        return self.exploration(action)

    def update(self):
        """Algorithm-specific training logic.

        The parameters are data to train the underlying model on. Algorithm-specific loss and optimization
        should be reflected here.
        """
        self.core_policy.update()

    def learn(self, experience_set: ExperienceSet):
        return self.core_policy.learn(experience_set)

    def set_state(self, policy_state):
        self.core_policy.set_state(policy_state)

    def load(self, path: str):
        self.core_policy.load(path)

    def save(self, path: str):
        self.core_policy.save(path)
