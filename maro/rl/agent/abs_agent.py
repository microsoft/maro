# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pickle
from abc import ABC, abstractmethod

from maro.rl.algorithms.abs_algorithm import AbsAlgorithm
from maro.rl.exploration.abs_explorer import AbsExplorer
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
        explorer (AbsExplorer): Explorer instance to generate exploratory actions. Defaults to None.
        experience_pool (AbsStore): It is used to store experiences processed by the experience shaper, which will be
            used by some value-based algorithms, such as DQN. Defaults to None.
    """
    def __init__(
        self,
        name: str,
        algorithm: AbsAlgorithm,
        explorer: AbsExplorer = None,
        experience_pool: AbsStore = None
    ):
        self._name = name
        self._algorithm = algorithm
        self._explorer = explorer
        self._experience_pool = experience_pool

    @property
    def algorithm(self):
        """Underlying algorithm employed by the agent."""
        return self._algorithm

    @property
    def explorer(self):
        """Explorer used by the agent to generate exploratory actions."""
        return self._explorer

    @property
    def experience_pool(self):
        """Underlying experience pool where the agent stores experiences."""
        return self._experience_pool

    def choose_action(self, model_state):
        """Choose an action using the underlying algorithm based on a preprocessed env state.

        Args:
            model_state: State vector as accepted by the underlying algorithm.
        Returns:
            Action given by the underlying policy model.
        """
        action_from_algorithm = self._algorithm.choose_action(model_state)
        return action_from_algorithm if self._explorer is None else self._explorer(action_from_algorithm)

    def load_exploration_params(self, exploration_params):
        if self._explorer:
            self._explorer.load_exploration_params(exploration_params)

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

    def load_models(self, *models, **model_dict):
        """Load models from memory."""
        self._algorithm.load_models(*models, **model_dict)

    def dump_models(self):
        """Return the algorithm's trainable models."""
        return self._algorithm.dump_models()

    def load_models_from_file(self, dir_path: str):
        """Load trainable models from disk.

        Load trainable models from the specified directory. The model file is always prefixed with the agent's name.

        Args:
            dir_path (str): path to the directory where the models are saved.
        """
        self._algorithm.load_models_from_file(os.path.join(dir_path, self._name))

    def dump_models_to_file(self, dir_path: str):
        """Dump the algorithm's trainable models to disk.

        Dump trainable models to the specified directory. The model file is always prefixed with the agent's name.

        Args:
            dir_path (str): path to the directory where the models are saved.
        """
        self._algorithm.dump_models_to_file(os.path.join(dir_path, self._name))

    def dump_experience_pool(self, dir_path: str):
        """Dump the experience pool to disk."""
        if self._experience_pool is not None:
            os.makedirs(dir_path, exist_ok=True)
            with open(os.path.join(dir_path, self._name), "wb") as fp:
                pickle.dump(self._experience_pool, fp)
