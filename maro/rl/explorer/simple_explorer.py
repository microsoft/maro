# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .abs_explorer import AbsExplorer


class LinearExplorer(AbsExplorer):
    """Exploration schedule where the exploration rate decreases with the number of episodes in a linear fashion.

    Args:
        max_eps (float): Maximum exploration rate, i.e., the exploration rate for the first episode.
        min_eps (float): Minimum exploration rate, i.e., the exploration rate for the last episode.
    """
    def __init__(self, max_eps: float, min_eps: float = .0):
        super().__init__()
        self._max_eps = max_eps
        self._min_eps = min_eps

    def generate_epsilon(self, current_ep, max_ep, performance_history=None):
        return self._min_eps + (self._max_eps - self._min_eps) * (1 - current_ep / (max_ep - 1))


class TwoPhaseLinearExplorer(AbsExplorer):
    """Exploration schedule that consists of two linear schedules separated by a split point.

    Args:
        start_eps (float): Exploration rate for the first episode.
        mid_eps (float): Exploration rate for the last episode.
        end_eps (float): The exploration rate where the switch from the first linear schedule to the second occurs.
            In other words, this is the exploration rate where the first linear schedule ends and the second begins.
        split_point (float): The point where the switch from the first linear schedule to the second occurs.
            Here "point" means the percentage of training loop completion, i.e., current_episode / max_episode,
            which means it must be a floating point number between 0 and 1.0.
    """
    def __init__(self, start_eps: float, mid_eps: float, end_eps: float, split_point: float):
        super().__init__()
        if split_point > 1.0 or split_point < 0.0:
            raise ValueError("split_point must be between 0 and 1.0")
        self._split_point = split_point
        self._start_eps = start_eps
        self._mid_eps = mid_eps
        self._end_eps = end_eps

    def generate_epsilon(self, current_ep, max_ep, performance_history=None):
        progress = current_ep / (max_ep - 1)
        if progress <= self._split_point:
            return self._start_eps - (self._start_eps - self._mid_eps) * progress / self._split_point
        else:
            return self._end_eps + (self._mid_eps - self._end_eps) * (1 - progress) / (1 - self._split_point)
