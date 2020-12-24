# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod

from maro.rl.models.learning_model import LearningModuleManager


class AbsAlgorithm(ABC):
    """Abstract RL algorithm class.

    The class provides uniform policy interfaces such as ``choose_action`` and ``train``. We also provide some
    predefined RL algorithm based on it, such DQN, A2C, etc. User can inherit from it to customize their own
    algorithms.

    Args:
        model (LearningModuleManager): Task model or container of task models required by the algorithm.
        config: Settings for the algorithm.
    """
    def __init__(self, model: LearningModuleManager, config):
        self._model = model
        self._config = config

    @property
    def model(self):
        return self._model

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

    @abstractmethod
    def train(self, *args, **kwargs):
        """Train models using samples.

        This method is algorithm-specific and needs to be implemented by the user. For example, for the DQN
        algorithm, this may look like train(self, state_batch, action_batch, reward_batch, next_state_batch).
        """
        return NotImplementedError

    def set_exploration_params(self, **params):
        pass
