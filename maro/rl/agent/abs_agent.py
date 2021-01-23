# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from abc import ABC, abstractmethod

import torch

from maro.rl.model.learning_model import AbsLearningModel
from maro.utils.exception.rl_toolkit_exception import UnrecognizedTask


class AbsAgent(ABC):
    """Abstract RL agent class.

    It's a sandbox for the RL algorithm. Scenario-specific details will be excluded.
    We focus on the abstraction algorithm development here. Environment observation and decision events will
    be converted to a uniform format before calling in. And the output will be converted to an environment
    executable format before return back to the environment. Its key responsibility is optimizing policy based
    on interaction with the environment.

    Args:
        name (str): Agent's name.
        model (AbsLearningModel): Task model or container of task models required by the algorithm.
        config: Settings for the algorithm.
        experience_pool (AbsStore): It is used to store experiences processed by the experience shaper, which will be
            used by some value-based algorithms, such as DQN. Defaults to None.
    """
    def __init__(self, name: str, model: AbsLearningModel, config, experience_pool=None):
        self._name = name
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model.to(self._device)
        self._config = config
        self._experience_pool = experience_pool

    @property
    def model(self):
        return self._model

    @property
    def experience_pool(self):
        """Underlying experience pool where the agent stores experiences."""
        return self._experience_pool

    @abstractmethod
    def choose_action(self, model_state):
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

    @abstractmethod
    def train(self, *args, **kwargs):
        """Training logic to be implemented by the user.

        For example, this may include drawing samples from the experience pool and the algorithm training on
        these samples.
        """
        return NotImplementedError

    def load_model(self, model):
        """Load models from memory."""
        self._model.load(model)

    def dump_model(self):
        """Return the algorithm's trainable models."""
        return self._model.dump()

    def load_model_from_file(self, dir_path: str):
        """Load trainable models from disk.

        Load trainable models from the specified directory. The model file is always prefixed with the agent's name.

        Args:
            dir_path (str): path to the directory where the models are saved.
        """
        self._model.load_from_file(os.path.join(dir_path, self._name))

    def dump_model_to_file(self, dir_path: str):
        """Dump the algorithm's trainable models to disk.

        Dump trainable models to the specified directory. The model file is always prefixed with the agent's name.

        Args:
            dir_path (str): path to the directory where the models are saved.
        """
        self._model.dump_to_file(os.path.join(dir_path, self._name))

    @staticmethod
    def validate_task_names(model_task_names, expected_task_names):
        task_names, expected_task_names = set(model_task_names), set(expected_task_names)
        if len(model_task_names) > 1 and task_names != expected_task_names:
            raise UnrecognizedTask(f"Expected task names {expected_task_names}, got {task_names}")
