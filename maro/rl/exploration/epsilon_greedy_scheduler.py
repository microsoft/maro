# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.utils.exception.rl_toolkit_exception import InvalidEpisodeError

from .abs_exploration_scheduler import AbsExplorationScheduler


class LinearEpsilonScheduler(AbsExplorationScheduler):
    """Linear exploration rate generator for epsilon-greedy exploration.

    Args:
        max_ep (int): Maximum number of episodes to run.
        start_eps (float): The exploration rate for the first episode.
        end_eps (float): The exploration rate for the last episode. Defaults to zero.

    """
    def __init__(self, max_ep: int, start_eps: float, end_eps: float = .0):
        if max_ep <= 0:
            raise InvalidEpisodeError("max_ep must be a positive integer.")
        super().__init__(max_ep)
        self._start_eps = start_eps
        self._end_eps = end_eps
        self._current_eps = start_eps
        self._eps_delta = (self._end_eps - self._start_eps) / (self._max_ep - 1)

    def __next__(self):
        if self._current_ep == self._max_ep:
            raise StopIteration
        eps = self._current_eps
        self._current_ep += 1
        self._current_eps += self._eps_delta
        return eps


class TwoPhaseLinearEpsilonScheduler(AbsExplorationScheduler):
    """Exploration schedule comprised of two linear schedules joined together for epsilon-greedy exploration.

    Args:
        max_ep (int): Maximum number of episodes to run.
        split_ep (float): The episode where the switch from the first linear schedule to the second occurs.
        start_eps (float): Exploration rate for the first episode.
        mid_eps (float): The exploration rate where the switch from the first linear schedule to the second occurs.
            In other words, this is the exploration rate where the first linear schedule ends and the second begins.
        end_eps (float): Exploration rate for the last episode. Defaults to zero.

    Returns:
        An iterator over the series of exploration rates from episode 0 to ``max_ep`` - 1.
    """
    def __init__(self, max_ep: int, split_ep: float, start_eps: float, mid_eps: float, end_eps: float = .0):
        if max_ep <= 0:
            raise InvalidEpisodeError("max_ep must be a positive integer.")
        if split_ep <= 0 or split_ep >= max_ep:
            raise ValueError("split_ep must be between 0 and max_ep - 1.")
        super().__init__(max_ep)
        self._split_ep = split_ep
        self._start_eps = start_eps
        self._mid_eps = mid_eps
        self._end_eps = end_eps
        self._current_eps = start_eps
        self._eps_delta_1 = (mid_eps - start_eps) / split_ep
        self._eps_delta_2 = (end_eps - mid_eps) / (max_ep - split_ep - 1)

    def __next__(self):
        if self._current_ep == self._max_ep:
            raise StopIteration
        eps = self._current_eps
        self._current_ep += 1
        self._current_eps += self._eps_delta_1 if self._current_ep < self._split_ep else self._eps_delta_2
        return eps
