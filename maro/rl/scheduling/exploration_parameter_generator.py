# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from typing import Union

import numpy as np


class StaticExplorationParameterGenerator(object):
    """Exploration parameter generator based on a pre-defined schedule.

    Args:
        max_ep (int): Maximum number of episodes to run.
    """
    def __init__(self, max_ep: int, **config):
        super().__init__()
        self._max_ep = max_ep
        self._config = config

    @abstractmethod
    def next(self):
        raise NotImplementedError


class DynamicExplorationParameterGenerator(object):
    """Dynamic exploration parameter generator based on the performances history."""
    def __init__(self, **config):
        super().__init__()
        self._config = config

    @abstractmethod
    def next(self, performance_history: list):
        raise NotImplementedError


class LinearExplorationParameterGenerator(StaticExplorationParameterGenerator):
    """Static exploration parameter generator based on a linear schedule.

    Args:
        max_ep (int): Maximum number of episodes to run.
        parameter_names ([str]): List of exploration parameter names.
        start_values (Union[float, list, tuple, np.ndarray]): Exploration parameter values for the first episode.
            These values must correspond to ``parameter_names``.
        end_values (Union[float, list, tuple, np.ndarray]): Exploration parameter values rate for the last episode.
            These values must correspond to ``parameter_names``.
    """
    def __init__(
        self,
        max_ep: int,
        *,
        parameter_names: [str],
        start_values: Union[float, list, tuple, np.ndarray],
        end_values: Union[float, list, tuple, np.ndarray]
    ):
        super().__init__(max_ep)
        self._parameter_names = parameter_names
        if isinstance(start_values, float):
            self._current_values = start_values * np.ones(len(self._parameter_names))
        elif isinstance(start_values, (list, tuple)):
            self._current_values = np.asarray(start_values)
        else:
            self._current_values = start_values

        if isinstance(end_values, float):
            end_values = end_values * np.ones(len(self._parameter_names))
        elif isinstance(end_values, (list, tuple)):
            end_values = np.asarray(end_values)

        self._delta = (end_values - self._current_values) / (max_ep - 1)

    def next(self):
        current_values = self._current_values.copy()
        self._current_values += self._delta
        return dict(zip(self._parameter_names, current_values))


class TwoPhaseLinearExplorationParameterGenerator(StaticExplorationParameterGenerator):
    """Exploration parameter generator based on two linear schedules joined together.

    Args:
        max_ep (int): Maximum number of episodes to run.
        parameter_names ([str]): List of exploration parameter names.
        split_ep (float): The episode where the switch from the first linear schedule to the second occurs.
        start_values (Union[float, list, tuple, np.ndarray]): Exploration parameter values for the first episode.
            These values must correspond to ``parameter_names``.
        mid_values (Union[float, list, tuple, np.ndarray]): Exploration parameter values where the switch from the
            first linear schedule to the second occurs. In other words, this is the exploration rate where the first
            linear schedule ends and the second begins. These values must correspond to ``parameter_names``.
        end_values (Union[float, list, tuple, np.ndarray]): Exploration parameter values for the last episode.
            These values must correspond to ``parameter_names``.

    Returns:
        An iterator over the series of exploration rates from episode 0 to ``max_ep`` - 1.
    """
    def __init__(
        self,
        max_ep: int,
        *,
        parameter_names: [str],
        split_ep: float,
        start_values: Union[float, list, tuple, np.ndarray],
        mid_values: Union[float, list, tuple, np.ndarray],
        end_values: Union[float, list, tuple, np.ndarray]
    ):
        if split_ep <= 0 or split_ep >= max_ep:
            raise ValueError("split_ep must be between 0 and max_ep - 1.")
        super().__init__(max_ep)
        self._parameter_names = parameter_names
        self._split_ep = split_ep
        if isinstance(start_values, float):
            self._current_values = start_values * np.ones(len(self._parameter_names))
        elif isinstance(start_values, (list, tuple)):
            self._current_values = np.asarray(start_values)
        else:
            self._current_values = start_values

        if isinstance(mid_values, float):
            mid_values = mid_values * np.ones(len(self._parameter_names))
        elif isinstance(mid_values, (list, tuple)):
            mid_values = np.asarray(mid_values)

        if isinstance(end_values, float):
            end_values = end_values * np.ones(len(self._parameter_names))
        elif isinstance(end_values, (list, tuple)):
            end_values = np.asarray(end_values)

        self._delta_1 = (mid_values - self._current_values) / split_ep
        self._delta_2 = (end_values - mid_values) / (max_ep - split_ep - 1)
        self._current_ep = 0

    def next(self):
        current_values = self._current_values.copy()
        self._current_values += self._delta_1 if self._current_ep < self._split_ep else self._delta_2
        self._current_ep += 1
        return dict(zip(self._parameter_names, current_values))
