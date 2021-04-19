# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod

import torch

from maro.rl.model import AbsCoreModel
from maro.rl.storage import SimpleStore


class AbsAlgorithm(ABC):
    """Abstract RL algorithm class.

    It's a sandbox for the RL algorithm. Scenario-specific details will be excluded.
    We focus on the abstraction algorithm development here. Environment observation and decision events will
    be converted to a uniform format before calling in. The output will be converted to an environment
    executable format before return back to the environment. Its key responsibility is optimizing policy based
    on interaction with the environment.

    Args:
        model (AbsCoreModel): Task model or container of task models required by the algorithm.
        config: Settings for the algorithm.
        experience_memory (SimpleStore): Experience memory for storing experiences
        empty_experience_memory_after_step (bool): If True, the experience memory will be emptied after each call
            to ``step``.
        min_new_experiences_to_trigger_learning (int): Minimum number of new experiences required to trigger learning.
            Defaults to 1.
        min_experiences_to_trigger_learning (int): Minimum number of experiences in the experience memory required for
            training. Defaults to 1.
    """
    def __init__(
        self,
        model: AbsCoreModel,
        config,
        experience_memory_size: int,
        experience_memory_overwrite_type: str,
        empty_experience_memory_after_step: bool,
        min_new_experiences_to_trigger_learning: int = 1,
        min_experiences_to_trigger_learning: int = 1
    ):
        self.model = model
        self.config = config
        self.experience_memory = SimpleStore(
            ["S", "A", "R", "S_"], capacity=experience_memory_size, overwrite_type=experience_memory_overwrite_type
        )
        self.empty_experience_memory_after_step = empty_experience_memory_after_step
        self.min_new_experiences_to_trigger_learning = min_new_experiences_to_trigger_learning
        self.min_experiences_to_trigger_learning = min_experiences_to_trigger_learning

    @abstractmethod
    def choose_action(self, state):
        """This method uses the underlying model(s) to compute an action from a shaped state.

        Args:
            state: A state object shaped by a ``StateShaper`` to conform to the model input format.

        Returns:
            The action to be taken given ``state``. It is usually necessary to use an ``ActionShaper`` to convert
            this to an environment executable action.
        """
        return NotImplementedError

    def set_exploration_params(self, **params):
        pass

    def learn(self, experiences: dict) -> bool:
        """Store experinces in the experience memory and train the model if necessary."""
        if set(experiences.keys()) != {"S", "A", "R", "S_"}:
            raise ValueError("The keys of experiences must be {'S', 'A', 'R', 'S_'}")
        self.experience_memory.put(experiences)
        if (
            len(experiences["S"]) >= self.min_new_experiences_to_trigger_learning and
            len(self.experience_memory) >= self.min_experiences_to_trigger_learning
        ):
            self.step()
            self._version_index += 1
            if self.empty_experience_memory_after_step:
                self.experience_memory.clear()
            return True
        return False

    @abstractmethod
    def step(self):
        """Algorithm-specific training logic.

        The parameters are data to train the underlying model on. Algorithm-specific loss and optimization
        should be reflected here.
        """
        return NotImplementedError
