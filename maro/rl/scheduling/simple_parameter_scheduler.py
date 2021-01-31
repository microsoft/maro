# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable, Union

import numpy as np

from .scheduler import Scheduler


class LinearParameterScheduler(Scheduler):
    """Static exploration parameter generator based on a linear schedule.

    Args:
        max_iter (int): Maximum number of iterations.
        parameter_names ([str]): List of exploration parameter names.
        start_values (Union[float, list, tuple, np.ndarray]): Exploration parameter values for the first episode.
            These values must correspond to ``parameter_names``.
        end_values (Union[float, list, tuple, np.ndarray]): Exploration parameter values rate for the last episode.
            These values must correspond to ``parameter_names``.
    """
    def __init__(
        self,
        max_iter: int,
        parameter_names: [str],
        start_values: Union[float, list, tuple, np.ndarray],
        end_values: Union[float, list, tuple, np.ndarray]
    ):
        super().__init__(max_iter)
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

        self._delta = (end_values - self._current_values) / (self._max_iter - 1)

    def next_params(self):
        current_values = self._current_values.copy()
        self._current_values += self._delta
        return dict(zip(self._parameter_names, current_values))


class TwoPhaseLinearParameterScheduler(Scheduler):
    """Exploration parameter generator based on two linear schedules joined together.

    Args:
        max_iter (int): Maximum number of episodes to run.
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
        An iterator over the series of exploration rates from episode 0 to ``max_iter`` - 1.
    """
    def __init__(
        self,
        max_iter: int,
        parameter_names: [str],
        split_ep: float,
        start_values: Union[float, list, tuple, np.ndarray],
        mid_values: Union[float, list, tuple, np.ndarray],
        end_values: Union[float, list, tuple, np.ndarray]
    ):
        if split_ep <= 0 or split_ep >= max_iter:
            raise ValueError("split_ep must be between 0 and max_iter - 1.")
        super().__init__(max_iter)
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
        self._delta_2 = (end_values - mid_values) / (max_iter - split_ep - 1)

    def next_params(self):
        current_values = self._current_values.copy()
        self._current_values += self._delta_1 if self._current_iter < self._split_ep else self._delta_2
        return dict(zip(self._parameter_names, current_values))
