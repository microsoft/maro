# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Union

import numpy as np

from .scheduler import Scheduler


class LinearParameterScheduler(Scheduler):
    """Static exploration parameter generator based on a linear schedule.

    Args:
        max_iter (int): Maximum number of iterations.
        parameter_names (List[str]): List of exploration parameter names.
        start (Union[float, list, tuple, np.ndarray]): Exploration parameter values for the first episode.
            These values must correspond to ``parameter_names``.
        end (Union[float, list, tuple, np.ndarray]): Exploration parameter values rate for the last episode.
            These values must correspond to ``parameter_names``.
    """
    def __init__(
        self,
        max_iter: int,
        parameter_names: List[str],
        start: Union[float, list, tuple, np.ndarray],
        end: Union[float, list, tuple, np.ndarray]
    ):
        super().__init__(max_iter)
        self._parameter_names = parameter_names
        if isinstance(start, float):
            self._current_values = start * np.ones(len(self._parameter_names))
        elif isinstance(start, (list, tuple)):
            self._current_values = np.asarray(start)
        else:
            self._current_values = start

        if isinstance(end, float):
            end = end * np.ones(len(self._parameter_names))
        elif isinstance(end, (list, tuple)):
            end = np.asarray(end)

        self._delta = (end - self._current_values) / (self._max_iter - 1)

    def next_params(self):
        current_values = self._current_values.copy()
        self._current_values += self._delta
        return dict(zip(self._parameter_names, current_values))


class TwoPhaseLinearParameterScheduler(Scheduler):
    """Exploration parameter generator based on two linear schedules joined together.

    Args:
        max_iter (int): Maximum number of iterations.
        parameter_names (List[str]): List of exploration parameter names.
        split (float): The point where the switch from the first linear schedule to the second occurs.
        start (Union[float, list, tuple, np.ndarray]): Exploration parameter values for the first episode.
            These values must correspond to ``parameter_names``.
        mid (Union[float, list, tuple, np.ndarray]): Exploration parameter values where the switch from the
            first linear schedule to the second occurs. In other words, this is the exploration rate where the first
            linear schedule ends and the second begins. These values must correspond to ``parameter_names``.
        end (Union[float, list, tuple, np.ndarray]): Exploration parameter values for the last episode.
            These values must correspond to ``parameter_names``.

    Returns:
        An iterator over the series of exploration rates from episode 0 to ``max_iter`` - 1.
    """
    def __init__(
        self,
        max_iter: int,
        parameter_names: List[str],
        split: float,
        start: Union[float, list, tuple, np.ndarray],
        mid: Union[float, list, tuple, np.ndarray],
        end: Union[float, list, tuple, np.ndarray]
    ):
        if split < 0 or split > 1.0:
            raise ValueError("split must be a float between 0 and 1.")
        super().__init__(max_iter)
        self._parameter_names = parameter_names
        self._split = int(self._max_iter * split)
        if isinstance(start, float):
            self._current_values = start * np.ones(len(self._parameter_names))
        elif isinstance(start, (list, tuple)):
            self._current_values = np.asarray(start)
        else:
            self._current_values = start

        if isinstance(mid, float):
            mid = mid * np.ones(len(self._parameter_names))
        elif isinstance(mid, (list, tuple)):
            mid = np.asarray(mid)

        if isinstance(end, float):
            end = end * np.ones(len(self._parameter_names))
        elif isinstance(end, (list, tuple)):
            end = np.asarray(end)

        self._delta_1 = (mid - self._current_values) / self._split
        self._delta_2 = (end - mid) / (max_iter - self._split - 1)

    def next_params(self):
        current_values = self._current_values.copy()
        self._current_values += self._delta_1 if self._iter_index < self._split else self._delta_2
        return dict(zip(self._parameter_names, current_values))
