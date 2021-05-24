# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod

from maro.rl.experience import ExperienceMemory


class AbsPolicy(ABC):
    """Abstract policy class."""
    def __init__(self):
        super().__init__()

    @abstractmethod
    def choose_action(self, state):
        raise NotImplementedError


class NullPolicy(AbsPolicy):
    """Dummy policy that does nothing.

    Note that the meaning of a "None" action may depend on the scenario.
    """
    def choose_action(self, state):
        return None


class AbsCorePolicy(AbsPolicy):
    """Policy that can update itself using simulation experiences.

    Reinforcement learning (RL) policies should inherit from this.

    Args:
        experience_memory (ExperienceMemory): An experience manager for storing and retrieving experiences
            for training.
    """
    def __init__(self, experience_memory: ExperienceMemory):
        super().__init__()
        self.experience_memory = experience_memory

    @abstractmethod
    def choose_action(self, state):
        raise NotImplementedError

    @abstractmethod
    def update(self):
        """Policy update logic is implemented here.

        This usually includes retrieving experiences as training samples from the experience manager and
        updating the underlying models using these samples.
        """
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

    def load(self, path: str):
        """Load the policy state from disk."""
        pass

    def save(self, path: str):
        """Save the policy state to disk."""
        pass
