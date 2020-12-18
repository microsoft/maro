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
    def __init__(self, last_k, threshold, measure_func: Callable[[list], Union[int, float]]):
        super().__init__(last_k, threshold)
        self._measure_func = measure_func

    def __call__(self, metric_series: list):
        if not self.is_triggered(metric_series):
            return False
        else:
            return self._measure_func(metric_series[-self._last_k:]) >= self._threshold


class RSDEarlyStoppingChecker(AbsEarlyStoppingChecker):
    """Early stopping checker based on the mean and standard deviation of the last k metric values."""
    def __call__(self, metric_series: list):
        if not self.is_triggered(metric_series):
            return False
        else:
            metric_series = metric_series[-self._last_k:]
            return stdev(metric_series) / mean(metric_series) < self._threshold


class MaxDeltaEarlyStoppingChecker(AbsEarlyStoppingChecker):
    """Early stopping checker based on the maximum relative variation over the last k metric values.

    The relative change is defined as |m(i+1) - m(i)| / m[i]. The maximum of the last k-1 changes in the metric series
    is compared with the threshold to determine if early stopping should be triggered.
    """
    def __call__(self, metric_series: list):
        if not self.is_triggered(metric_series):
            return False
        else:
            metric_series = metric_series[-self._last_k:]
            max_delta = max(abs(val2 - val1) / val1 for val1, val2 in zip(metric_series, metric_series[1:]))
            return max_delta < self._threshold
