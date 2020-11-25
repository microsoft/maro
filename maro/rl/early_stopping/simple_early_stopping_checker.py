# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from statistics import mean, stdev
from typing import Callable, Union

from .abs_early_stopping_checker import AbsEarlyStoppingChecker


class SimpleEarlyStoppingChecker(AbsEarlyStoppingChecker):
    """Early stopping checker based on the some simple measure.

    The measure is obtained by applying a user-defined measure function to the last k metric values. The measure
    function must take a list as input and output a single number.
    """
    def __init__(
        self,
        last_k: int,
        threshold: float,
        warmup_ep: int,
        metric_func: Callable,
        measure_func: Callable[[list], Union[int, float]]
    ):
        super().__init__(last_k, threshold, warmup_ep, metric_func)
        self._measure_func = measure_func

    def check(self):
        return self._measure_func(self._metric_series) >= self._threshold


class RSDEarlyStoppingChecker(AbsEarlyStoppingChecker):
    """Early stopping checker based on the mean and standard deviation of the last k metric values."""
    def check(self):
        return stdev(self._metric_series) / mean(self._metric_series) < self._threshold


class MaxDeltaEarlyStoppingChecker(AbsEarlyStoppingChecker):
    """Early stopping checker based on the maximum relative variation over the last k metric values.

    The relative change is defined as |m(i+1) - m(i)| / m[i]. The maximum of the last k-1 changes in the metric series
    is compared with the threshold to determine if early stopping should be triggered.
    """
    def check(self):
        max_delta = max(
            abs(self._metric_series[i] - self._metric_series[i - 1]) / self._metric_series[i - 1]
            for i in range(1, self._last_k)
        )
        return max_delta < self._threshold
