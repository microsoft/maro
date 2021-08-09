# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod

from maro.rl.experience import ExperienceSet
from maro.rl.exploration import AbsExploration


class AbsAlgorithm(ABC):
    """Abstract algorithm.

    Reinforcement learning (RL) algorithms should inherit from this.

    Args:
        exploration (AbsExploration): Exploration strategy for generating exploratory actions. Defaults to None.
    """
    def __init__(self, exploration: AbsExploration = None):
        super().__init__()
        self.exploration = exploration

    @abstractmethod
    def choose_action(self, state, explore: bool = False):
        raise NotImplementedError

    @abstractmethod
    def apply(self, grad_dict: dict):
        """Update the underlying parameters (i.e., network weights) with gradients."""
        pass

    @abstractmethod
    def learn(self, experience_batch: ExperienceSet, inplace: bool = True) -> tuple:
        """Update logic is implemented here."""
        raise NotImplementedError

    @abstractmethod
    def get_state(self, inference: bool = True):
        """Return the current state of the policy.

        The implementation must go hand in hand with that of ``set_state``. For example, if a torch model
        is contained in the policy, ``get_state`` may include a call to ``state_dict()`` on the model, while
        ``set_state`` should accordingly include ``load_state_dict()``.

        Args:
            learning (bool): If True, the returned state is for inference purpose only. This parameter
                may be ignored for some algorithms.
        """
        pass

    @abstractmethod
    def set_state(self, policy_state):
        """Set the policy state to ``policy_state``.

        The implementation must be in correspondence with that of ``get_state``. For example, if a torch model
        is contained in the policy, ``set_state`` may include a call to ``load_state_dict()`` on the model, while
        ``get_state`` should accordingly include ``state_dict()``.
        """
        pass

    def load(self, path: str):
        """Load the policy state from disk."""
        pass

    def save(self, path: str):
        """Save the policy state to disk."""
        pass
