# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from statistics import mean, stdev

from .abs_early_stopping_checker import AbsEarlyStoppingChecker


class RSDEarlyStoppingChecker(AbsEarlyStoppingChecker):
    """Simple early stopping checker based on the mean and standard deviation of the last k performance records.

    Args:
        last_k (int): Number of the latest performance records to check for early stopping.
        threshold (float): The threshold value against which the early stopping metric is compared. The early stopping
            condition is satisfied if the metric obtained using the ``performance_metric_func`` is below this threshold.
    """
    def __init__(self, last_k: int, threshold: float):
        super().__init__()
        self._last_k = last_k
        self._threshold = threshold

    def __call__(self, metric_series):
        if self._last_k > len(metric_series):
            return False
        else:
            return stdev(metric_series) / mean(metric_series) < self._threshold


class MaxDeltaEarlyStoppingChecker(AbsEarlyStoppingChecker):
    """Simple early stopping checker based on the mean and standard deviation of the last k performance records.

    Args:
        last_k (int): Number of the latest performance records to check for early stopping.
        threshold (float): The threshold value against which the early stopping metric is compared. The early stopping
            condition is satisfied if the metric obtained using the ``performance_metric_func`` is below this threshold.
    """
    def __init__(self, last_k: int, threshold: float):
        super().__init__()
        self._last_k = last_k
        self._threshold = threshold

    def __call__(self, metric_series):
        if self._last_k > len(metric_series):
            return False
        else:
            metric_series = metric_series[-self._last_k:]
            max_delta = max(abs(val2 - val1) / val1 for val1, val2 in zip(metric_series, metric_series[1:]))
            return max_delta < self._threshold
