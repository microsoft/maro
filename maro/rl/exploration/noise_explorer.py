# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from typing import Union

import numpy as np

from maro.utils.exception.rl_toolkit_exception import MissingExplorationParametersError

from .abs_explorer import AbsExplorer


class NoiseExplorer(AbsExplorer):
    """Explorer that adds a random noise to a model-generated action."""
    def __init__(
        self,
        action_dim: int,
        min_action: Union[float, np.ndarray] = None,
        max_action: Union[float, np.ndarray] = None
    ):
        super().__init__()
        self._action_dim = action_dim
        self._min_action = min_action
        self._max_action = max_action

    @abstractmethod
    def load_exploration_params(self, **exploration_params):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, action):
        return NotImplementedError


class UniformNoiseExplorer(NoiseExplorer):
    """Explorer that adds a random noise to a model-generated action sampled from a uniform distribution."""
    def __init__(
        self,
        action_dim: int,
        min_action: Union[float, np.ndarray] = None,
        max_action: Union[float, np.ndarray] = None
    ):
        super().__init__(action_dim, min_action, max_action)
        self._noise_bound = None

    def load_exploration_params(self, *, noise_bound: float):
        self._noise_bound = noise_bound

    def __call__(self, action: np.ndarray):
        if self._noise_bound is None:
            raise MissingExplorationParametersError(
                'Noise bound is not set. Use load_exploration_params with keyword argument "noise_bound" to '
                'load the exploration parameters first.'
            )
        action += np.random.uniform(-self._noise_bound, self._noise_bound, self._action_dim)
        if self._min_action is not None or self._max_action is not None:
            return np.clip(action, self._min_action, self._max_action)
        else:
            return action


class GaussianNoiseExplorer(NoiseExplorer):
    """Explorer that adds a random noise to a model-generated action sampled from a Gaussian distribution."""
    def __init__(
        self,
        action_dim: int,
        min_action: Union[float, np.ndarray] = None,
        max_action: Union[float, np.ndarray] = None
    ):
        super().__init__(action_dim, min_action, max_action)
        self._noise_scale = None

    def load_exploration_params(self, *, noise_scale: float):
        self._noise_scale = noise_scale

    def __call__(self, action: np.ndarray):
        if self._noise_scale is None:
            raise MissingExplorationParametersError(
                'Noise scale is not set. Use load_exploration_params with keyword argument "noise_scale" to '
                'load the exploration parameters first.'
            )
        action += np.random.normal(scale=self._noise_scale, size=self._action_dim)
        if self._min_action is not None or self._max_action is not None:
            return np.clip(action, self._min_action, self._max_action)
        else:
            return action
