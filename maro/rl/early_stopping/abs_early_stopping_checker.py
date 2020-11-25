# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod
from collections import deque
from typing import Callable


class AbsEarlyStoppingChecker(ABC):
    """Class that checks for early stopping conditions.

    Args:
        last_k (int): Number of the latest metric values to check for early stopping.
        threshold (float): The threshold value against which a user-defined measure is compared to determine
            whether early-stopping should be triggered.
        warmup_ep (int): Number of episodes before early stopping checker takes effect.
        metric_func (Callable): Function to extract early stopping metric from a performance record.
    """
    def __init__(self, last_k: int, threshold: float, warmup_ep: int, metric_func: Callable):
        super().__init__()
        self._last_k = last_k
        self._threshold = threshold
        self._metric_func = metric_func
        self._warmup_ep = warmup_ep
        self._metric_series = deque()
        self._ep_count = 0

    def update(self, performance) -> bool:
        """Update with the latest performance record and check whether an early stopping condition is met.

        Args:
            performance: Performance record from the latest roll-out episode.

        Returns:
            A boolean value indicating whether early stopping should be triggered.
        """
        self._ep_count += 1
        if isinstance(performance, list):
            self._metric_series.extend(map(self._metric_func, (perf for _, perf in performance)))
        else:
            self._metric_series.append(self._metric_func(performance))
        if self._ep_count < self._warmup_ep or len(self._metric_series) < self._last_k:
            return False

        while len(self._metric_series) > self._last_k:
            self._metric_series.popleft()

        return self.check()

    @abstractmethod
    def check(self) -> bool:
        return NotImplemented

    def __or__(self, other_checker):
        """Return a checker that is the result of logical OR between itself and another checker.

        The resulting checker returns True iff at least one of the checkers returns True.
        """
        class OrChecker:
            def __init__(self, checker, other):
                self._checker = checker
                self._other_checker = other

            def push(self, performance) -> bool:
                return self._checker.update(performance) or self._other_checker.update(performance)

        return OrChecker(self, other_checker)

    def __and__(self, other_checker):
        """Return a checker that is the result of logical AND between itself and another checker.

        The resulting checker returns True iff both checkers return True.
        """
        class AndChecker:
            def __init__(self, checker, other):
                super().__init__()
                self._checker = checker
                self._other_checker = other

            def push(self, performance) -> bool:
                result = self._checker.update(performance)
                other_result = self._other_checker.update(performance)
                return result and other_result

        return AndChecker(self, other_checker)

    def __xor__(self, other_checker):
        """Return a checker that is the result of logical XOR between itself and another checker.

        The resulting checker returns True iff one checker returns True and the other returns False.
        """
        class XorChecker:
            def __init__(self, checker, other):
                self._checker = checker
                self._other_checker = other

            def __call__(self, performance) -> bool:
                return self._checker.update(performance) ^ self._other_checker.update(performance)

        return XorChecker(self, other_checker)

    def __invert__(self):
        """Return a checker that is the result of logical NOT of itself.

        The resulting checker returns True iff itself returns False.
        """
        class InverseChecker:
            def __init__(self, checker):
                super().__init__()
                self._checker = checker

            def __call__(self, performance) -> bool:
                return not self._checker.update(performance)

        return InverseChecker(self)
