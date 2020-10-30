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
