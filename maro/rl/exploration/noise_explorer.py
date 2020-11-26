# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from typing import Union

import numpy as np

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
    def update(self, **exploration_params):
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
        max_action: Union[float, np.ndarray] = None,
        noise_lower_bound: Union[float, np.ndarray] = .0,
        noise_upper_bound: Union[float, np.ndarray] = .0
    ):
        super().__init__(action_dim, min_action, max_action)
        self._noise_lower_bound = noise_lower_bound
        self._noise_upper_bound = noise_upper_bound

    def update(self, *, noise_lower_bound: Union[float, np.ndarray], noise_upper_bound: Union[float, np.ndarray]):
        self._noise_lower_bound = noise_lower_bound
        self._noise_upper_bound = noise_upper_bound

    def __call__(self, action: np.ndarray):
        action += np.random.uniform(self._noise_lower_bound, self._noise_upper_bound, self._action_dim)
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
        max_action: Union[float, np.ndarray] = None,
        noise_mean: Union[float, np.ndarray] = .0,
        noise_stddev: Union[float, np.ndarray] = .0,
        is_relative: bool = False
    ):
        super().__init__(action_dim, min_action, max_action)
        if is_relative and noise_mean != .0:
            raise ValueError("Standard deviation cannot be relative if noise mean is non-zero.")
        self._noise_mean = noise_mean
        self._noise_stddev = noise_stddev
        self._is_relative = is_relative

    def update(self, *, noise_mean: Union[float, np.ndarray], noise_stddev: Union[float, np.ndarray]):
        self._noise_mean = noise_mean
        self._noise_stddev = noise_stddev

    def __call__(self, action: np.ndarray):
        noise = np.random.normal(loc=self._noise_mean, scale=self._noise_stddev, size=self._action_dim)
        action += (noise * action) if self._is_relative else noise
        if self._min_action is not None or self._max_action is not None:
            return np.clip(action, self._min_action, self._max_action)
        else:
            return action
