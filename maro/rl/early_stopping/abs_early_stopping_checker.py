# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod


class AbsEarlyStoppingChecker(ABC):
    """Class that checks for early stopping conditions.

    Implementations of this abstract class usually involve user-defined early stopping conditions.
    """
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, metric_series) -> bool:
        """Check whether the early stopping condition (defined in the class) is met.

        Args:
            metric_series: History of performances (from actors) used to check whether the early stopping
                condition is satisfied.

        Returns:
            A boolean value indicating whether early stopping should be triggered.
        """
        return NotImplemented

    def __or__(self, other_checker):
        """Return a checker that is the result of logical OR between itself and another checker.

        The resulting checker returns True iff at least one of the checkers returns True.
        """
        class OrChecker(AbsEarlyStoppingChecker):
            def __init__(self, checker, other):
                super().__init__()
                self._checker = checker
                self._other_checker = other

            def __call__(self, metric_series) -> bool:
                return self._checker(metric_series) or self._other_checker(metric_series)

        return OrChecker(self, other_checker)

    def __and__(self, other_checker):
        """Return a checker that is the result of logical AND between itself and another checker.

        The resulting checker returns True iff both checkers return True.
        """
        class AndChecker(AbsEarlyStoppingChecker):
            def __init__(self, checker, other):
                super().__init__()
                self._checker = checker
                self._other_checker = other

            def __call__(self, metric_series) -> bool:
                return self._checker(metric_series) and self._other_checker(metric_series)

        return AndChecker(self, other_checker)

    def __xor__(self, other_checker):
        """Return a checker that is the result of logical XOR between itself and another checker.

        The resulting checker returns True iff one checker returns True and the other returns False.
        """
        class XorChecker(AbsEarlyStoppingChecker):
            def __init__(self, checker, other):
                super().__init__()
                self._checker = checker
                self._other_checker = other

            def __call__(self, metric_series) -> bool:
                return self._checker(metric_series) ^ self._other_checker(metric_series)

        return XorChecker(self, other_checker)

    def __invert__(self):
        """Return a checker that is the result of logical NOT of itself.

        The resulting checker returns True iff itself returns False.
        """
        class NotChecker(AbsEarlyStoppingChecker):
            def __init__(self, checker):
                super().__init__()
                self._checker = checker

            def __call__(self, metric_series) -> bool:
                return not self._checker(metric_series)

        return NotChecker(self)
