# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from statistics import mean, stdev

from .abs_early_stopping_checker import AbsEarlyStoppingChecker


class MeanValueChecker(AbsEarlyStoppingChecker):
    """Early stopping checker based on the mean of the last k metric values.

    Args:
        last_k (int): Number of the latest metric values to check for early stopping.
        threshold (float): The threshold value against which the mean of the ``last_k`` metric values is compared.
            The early stopping condition is hit if the mean is greater than or equal to this threshold.
    """
    def __init__(self, last_k, threshold):
        super().__init__()
        self._last_k = last_k
        self._threshold = threshold

    def __call__(self, metric_series: list):
        if self._last_k > len(metric_series):
            return False
        else:
            return mean(metric_series[-self._last_k:]) >= self._threshold


class RSDEarlyStoppingChecker(AbsEarlyStoppingChecker):
    """Early stopping checker based on the mean and standard deviation of the last k metric values.

    Args:
        last_k (int): Number of the latest metric values to check for early stopping.
        threshold (float): The threshold value against which the relative standard deviation (RSD) of the ``last_k``
            metric values is compared. The early stopping condition is hit if the RSD is below this threshold.
    """
    def __init__(self, last_k: int, threshold: float):
        super().__init__()
        self._last_k = last_k
        self._threshold = threshold

    def __call__(self, metric_series: list):
        if self._last_k > len(metric_series):
            return False
        else:
            metric_series = metric_series[-self._last_k:]
            return stdev(metric_series) / mean(metric_series) < self._threshold


class MaxDeltaEarlyStoppingChecker(AbsEarlyStoppingChecker):
    """Early stopping checker based on the maximum relative variation over the last k metric values.

    The relative change is defined as |m(i+1) - m(i)| / m[i]. The maximum of the last k-1 changes in the metric series
    is compared with the threshold to determine if early stopping should be triggered.

    Args:
        last_k (int): Number of the latest metric values to check for early stopping.
        threshold (float): The threshold value against which the maximum relative variation (MRV) of the ``last_k``
            metric values is compared. The early stopping condition is hit if the MRV is below this threshold.
    """
    def __init__(self, last_k: int, threshold: float):
        super().__init__()
        self._last_k = last_k
        self._threshold = threshold

    def __call__(self, metric_series: list):
        if self._last_k > len(metric_series):
            return False
        else:
            metric_series = metric_series[-self._last_k:]
            max_delta = max(abs(val2 - val1) / val1 for val1, val2 in zip(metric_series, metric_series[1:]))
            return max_delta < self._threshold
