# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .abs_explorer import AbsExplorer


class LinearExplorer(AbsExplorer):
    """A simple linear exploration scheme."""
    def __init__(self, max_eps: float, min_eps: float = .0):
        super().__init__()
        self._max_eps = max_eps
        self._min_eps = min_eps

    def generate_epsilon(self, current_ep, max_ep, performance_history=None):
        return self._min_eps + (self._max_eps - self._min_eps) * (1 - current_ep / max_ep)


class TwoPhaseLinearExplorer(AbsExplorer):
    """An exploration scheme that consists of two linear schedules separated by a split point."""
    def __init__(self, progress_split: float, eps_split: float, max_eps: float, min_eps: float = 0):
        super().__init__()
        self._progress_split = progress_split
        self._eps_split = eps_split
        self._max_eps = max_eps
        self._min_eps = min_eps

    def generate_epsilon(self, current_ep, max_ep, performance_history=None):
        progress = current_ep / max_ep
        if progress <= self._progress_split:
            return self._max_eps - (self._max_eps - self._eps_split) * progress / self._progress_split
        else:
            return self._min_eps + (self._eps_split - self._min_eps) * (1 - progress) / (1 - self._progress_split)
