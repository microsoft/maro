# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from typing import Union

import numpy as np

from .abs_exploration import AbsExploration


class NoiseExploration(AbsExploration):
    """Exploration that adds a random noise to a model-generated action."""
    def __init__(
        self,
        min_action: Union[float, list, np.ndarray] = None,
        max_action: Union[float, list, np.ndarray] = None
    ):
        if isinstance(min_action, (list, np.ndarray)) and isinstance(max_action, (list, np.ndarray)):
            assert len(min_action) == len(max_action), "min_action and max_action should have the same dimension."
        super().__init__()
        self.min_action = min_action
        self.max_action = max_action

    @abstractmethod
    def __call__(self, action) -> np.ndarray:
        raise NotImplementedError


class UniformNoiseExploration(NoiseExploration):
    """Exploration that adds a random noise to a model-generated action sampled from a uniform distribution."""
    def __init__(
        self,
        min_action: Union[float, list, np.ndarray] = None,
        max_action: Union[float, list, np.ndarray] = None,
        noise_lower_bound: Union[float, list, np.ndarray] = .0,
        noise_upper_bound: Union[float, list, np.ndarray] = .0
    ):
        if isinstance(noise_upper_bound, (list, np.ndarray)) and isinstance(noise_upper_bound, (list, np.ndarray)):
            assert len(noise_lower_bound) == len(noise_upper_bound), \
                "noise_lower_bound and noise_upper_bound should have the same dimension."
        super().__init__(min_action, max_action)
        self.noise_lower_bound = noise_lower_bound
        self.noise_upper_bound = noise_upper_bound

    def __call__(self, action: np.ndarray) -> np.ndarray:
        return np.array([self._get_exploration_action(act) for act in action])

    def _get_exploration_action(self, action):
        action += np.random.uniform(self.noise_lower_bound, self.noise_upper_bound)
        if self.min_action is not None or self.max_action is not None:
            return np.clip(action, self.min_action, self.max_action)
        else:
            return action


class GaussianNoiseExploration(NoiseExploration):
    """Exploration that adds a random noise to a model-generated action sampled from a Gaussian distribution."""
    def __init__(
        self,
        min_action: Union[float, list, np.ndarray] = None,
        max_action: Union[float, list, np.ndarray] = None,
        noise_mean: Union[float, list, np.ndarray] = .0,
        noise_stddev: Union[float, list, np.ndarray] = .0,
        is_relative: bool = False
    ):
        if isinstance(noise_mean, (list, np.ndarray)) and isinstance(noise_stddev, (list, np.ndarray)):
            assert len(noise_mean) == len(noise_stddev), "noise_mean and noise_stddev should have the same dimension."
        if is_relative and noise_mean != .0:
            raise ValueError("Standard deviation cannot be relative if noise mean is non-zero.")
        super().__init__(min_action, max_action)
        self.noise_mean = noise_mean
        self.noise_stddev = noise_stddev
        self.is_relative = is_relative

    def __call__(self, action: np.ndarray) -> np.ndarray:
        return np.array([self._get_exploration_action(act) for act in action])

    def _get_exploration_action(self, action):
        noise = np.random.normal(loc=self.noise_mean, scale=self.noise_stddev)
        action += (noise * action) if self.is_relative else noise
        if self.min_action is not None or self.max_action is not None:
            return np.clip(action, self.min_action, self.max_action)
        else:
            return action
