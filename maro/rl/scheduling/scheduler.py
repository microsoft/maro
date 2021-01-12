# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable

from maro.utils.exception.rl_toolkit_exception import InfiniteTrainingLoop, InvalidEpisode


class Scheduler(object):
    """Scheduler that generates exploration parameters for each episode.

    Args:
        max_ep (int): Maximum number of episodes to be run. If -1, an early stopping callback is expected to prevent
            the training loop from running forever.
        early_stopping_checker (Callable): Function that returns a boolean indicating whether early stopping should
            be triggered. Defaults to None, in which case no early stopping check will be performed.
    """

    def __init__(self, max_ep: int, early_stopping_checker: Callable = None):
        if max_ep < -1:
            raise InvalidEpisode("max_episode can only be a non-negative integer or -1.")
        if max_ep == -1 and early_stopping_checker is None:
            raise InfiniteTrainingLoop(
                "A positive max_ep or an early stopping checker must be provided to prevent the training loop from "
                "running forever."
            )
        self._max_ep = max_ep
        self._early_stopping_checker = early_stopping_checker
        self._current_ep = -1
        self._performance_history = []
        self._exploration_params = None

    def __iter__(self):
        return self

    def __next__(self):
        self._current_ep += 1
        if self._current_ep == self._max_ep:
            raise StopIteration
        if self._early_stopping_checker and self._early_stopping_checker(self._performance_history):
            raise StopIteration

        self._exploration_params = self.get_next_exploration_params()
        return self._exploration_params

    def get_next_exploration_params(self):
        pass

    @property
    def current_ep(self):
        return self._current_ep

    @property
    def exploration_params(self):
        return self._exploration_params

    def record_performance(self, performance):
        self._performance_history.append(performance)

    def reset(self):
        self._current_ep = -1
        self._performance_history = []
        self._exploration_params = None
