# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from abc import ABC, abstractmethod

from maro.rl.algorithms.abs_algorithm import AbsAlgorithm
from maro.rl.storage.abs_store import AbsStore


class AbsAgent(ABC):
    """Abstract RL agent class.

    It's a sandbox for the RL algorithm. Scenario-specific details will be excluded.
    We focus on the abstraction algorithm development here. Environment observation and decision events will
    be converted to a uniform format before calling in. And the output will be converted to an environment
    executable format before return back to the environment. Its key responsibility is optimizing policy based
    on interaction with the environment.

    Args:
        name (str): Agent's name.
        algorithm (AbsAlgorithm): A concrete algorithm instance that inherits from AbstractAlgorithm.
            This is the centerpiece of the Agent class and is responsible for the most important tasks of an agent:
            choosing actions and optimizing models.
        experience_pool (AbsStore): It is used to store experiences processed by the experience shaper, which will be
            used by some value-based algorithms, such as DQN. Defaults to None.
    """
    def __init__(
        self,
        name: str,
        algorithm: AbsAlgorithm,
        experience_pool: AbsStore = None
    ):
        self._name = name
        self._algorithm = algorithm
        self._experience_pool = experience_pool

    @property
    def algorithm(self):
        """Underlying algorithm employed by the agent."""
        return self._algorithm

    @property
    def experience_pool(self):
        """Underlying experience pool where the agent stores experiences."""
        return self._experience_pool

    def choose_action(self, model_state):
        """Choose an action using the underlying algorithm based on a preprocessed env state.

        Args:
            model_state: State vector as accepted by the underlying algorithm.
        Returns:
            If the agent's explorer is None, the action given by the underlying model is returned. Otherwise,
            an exploratory action is returned.
        """
        return self._algorithm.choose_action(model_state)

    def set_exploration_params(self, **params):
        self._algorithm.set_exploration_params(**params)

    @abstractmethod
    def train(self, *args, **kwargs):
        """Training logic to be implemented by the user.

        For example, this may include drawing samples from the experience pool and the algorithm training on
        these samples.
        """
        return NotImplementedError

    def store_experiences(self, experiences):
        """Store new experiences in the experience pool."""
        if self._experience_pool is not None:
            self._experience_pool.put(experiences)

    def load_model(self, model):
        """Load models from memory."""
        self._algorithm.model.load(model)

    def dump_model(self):
        """Return the algorithm's trainable models."""
        return self._algorithm.model.dump()

    def load_model_from_file(self, dir_path: str):
        """Load trainable models from disk.

        Load trainable models from the specified directory. The model file is always prefixed with the agent's name.

        Args:
            dir_path (str): path to the directory where the models are saved.
        """
        self._algorithm.model.load_from_file(os.path.join(dir_path, self._name))

    def dump_model_to_file(self, dir_path: str):
        """Dump the algorithm's trainable models to disk.

        Dump trainable models to the specified directory. The model file is always prefixed with the agent's name.

        Args:
            dir_path (str): path to the directory where the models are saved.
        """
        self._algorithm.model.dump_to_file(os.path.join(dir_path, self._name))
