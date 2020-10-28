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
        progress_split (float): The point where the switch from the first linear schedule to the second occurs.
            Here "point" means the percentage of training loop completion, i.e., current_episode / max_episode,
            which means it must be a floating point number between 0 and 1.0.
        eps_split (float): The exploration rate where the switch from the first linear schedule to the second occurs.
            In other words, this is the exploration rate where the first linear schedule ends and the second begins.
        max_eps (float): Maximum exploration rate, i.e., the exploration rate for the first episode.
        min_eps (float): Minimum exploration rate, i.e., the exploration rate for the last episode.
    """
    def __init__(self, progress_split: float, eps_split: float, max_eps: float, min_eps: float = 0):
        super().__init__()
        if progress_split > 1.0 or progress_split < 0.0:
            raise ValueError("progress_split must be between 0 and 1.0")
        self._progress_split = progress_split
        self._eps_split = eps_split
        self._max_eps = max_eps
        self._min_eps = min_eps

    def generate_epsilon(self, current_ep, max_ep, performance_history=None):
        progress = current_ep / (max_ep - 1)
        if progress <= self._progress_split:
            return self._max_eps - (self._max_eps - self._eps_split) * progress / self._progress_split
        else:
            return self._min_eps + (self._eps_split - self._min_eps) * (1 - progress) / (1 - self._progress_split)
