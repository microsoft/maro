# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod


class AbsExplorationScheduler(ABC):
    """Scheduler that generates exploration parameters for each episode.

    Args:
        max_ep (int): Maximum number of episodes to be run.
    """
    def __init__(self, max_ep: int):
        self._max_ep = max_ep
        self._current_ep = 0

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self):
        raise NotImplementedError

    @property
    def current_ep(self):
        return self._current_ep


class NullExplorationScheduler(AbsExplorationScheduler):
    """Dummy scheduler that generates."""
    def __next__(self):
        if self._current_ep == self._max_ep:
            raise StopIteration
        self._current_ep += 1
