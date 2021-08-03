# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from typing import Callable

from maro.rl.experience import ExperienceSet
from maro.rl.exploration import AbsExploration


class AbsAlgorithm(ABC):
    """Policy that can update itself using simulation experiences.

    Reinforcement learning (RL) policies should inherit from this.

    Args:
        exploration (AbsExploration): Exploration strategy for generating exploratory actions. Defaults to None.
        post_step (Callable): Custom function to be called after each gradient step. This can be used for tracking
            the learning progress. The function should have signature (loss, tracker) -> None. Defaults to None.
    """
    def __init__(self, exploration: AbsExploration = None, post_learn: Callable = None):
        super().__init__()
        self.exploration = exploration
        self.exploring = True

    @abstractmethod
    def choose_action(self, state, explore: bool = False):
        raise NotImplementedError

    def get_update_info(self, experience_batch: ExperienceSet):
        pass

    def apply(self):
        pass

    @abstractmethod
    def learn(self, experience_batch: ExperienceSet):
        """Update logic is implemented here."""
        raise NotImplementedError

    @abstractmethod
    def get_state(self):
        """Return the current state of the policy.

        The implementation must be in correspondence with that of ``set_state``. For example, if a torch model
        is contained in the policy, ``get_state`` may include a call to ``state_dict()`` on the model, while
        ``set_state`` should accordingly include ``load_state_dict()``.
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

    def explore(self):
        self.exploring = False

    def exploit(self):
        self.exploring = True

    def exploration_step(self):
        if self.exploration:
            self.exploration.step()

    def load(self, path: str):
        """Load the policy state from disk."""
        pass

    def save(self, path: str):
        """Save the policy state to disk."""
        pass
