# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
from abc import ABC, abstractmethod

from maro.rl.experience import ExperienceManager, ExperienceSet


class AbsPolicy(ABC):
    """Abstract policy class."""
    def __init__(self, name: str):
        self._name = name
        super().__init__()

    @property
    def name(self):
        return self._name

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
        name (str): Policy name.
        experience_manager (ExperienceManager): An experience manager for storing and retrieving experiences
            for training.
        update_trigger (int): Minimum number of new experiences required to trigger an ``update`` call. Defaults to 1.
        warmup (int): Minimum number of experiences in the experience memory required to trigger an ``update`` call.
            Defaults to 1.
    """
    def __init__(
        self,
        name: str,
        experience_manager: ExperienceManager,
        update_trigger: int = 1,
        warmup: int = 1
    ):
        super().__init__(name)
        self.experience_manager = experience_manager
        self.update_trigger = update_trigger
        self.warmup = warmup
        self._new_exp_counter = 0

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

    def on_experiences(self, exp: ExperienceSet) -> bool:
        self.experience_manager.put(exp)
        self._new_exp_counter += exp.size
        print(
            f"Policy {self._name}: exp mem size = {self.experience_manager.size}, incoming: {exp.size}, "
            f"new exp = {self._new_exp_counter}"
        )
        if self.experience_manager.size >= self.warmup and self._new_exp_counter >= self.update_trigger:
            t0 = time.time()
            self.update()
            print(f"policy update time: {time.time() - t0}")
            self._new_exp_counter = 0
            return True

        return False

    def load(self, path: str):
        """Load the policy state from disk."""
        pass

    def save(self, path: str):
        """Save the policy state to disk."""
        pass
