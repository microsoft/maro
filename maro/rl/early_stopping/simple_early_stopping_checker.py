# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable

from statistics import mean, stdev

from .abs_early_stopping_checker import AbsEarlyStoppingChecker


class SimpleEarlyStoppingChecker(AbsEarlyStoppingChecker):
    """Simple early stopping checker based on the mean and standard deviation of the last k performance records.

    Args:
        last_k (int): Number of the latest performance records to check for early stopping.
        metric_func (Callable): A function to obtain the metric from a performance record to be evaluated against a
            threshold value.
        threshold (float): The threshold value against which the early stopping metric is compared. The early stopping
            condition is satisfied if the metric obtained using the ``performance_metric_func`` is below this threshold.
    """
    def __init__(self, last_k: int, metric_func: Callable, threshold: float):
        super().__init__()
        self._last_k = last_k
        self._metric_func = metric_func
        self._threshold = threshold

    def __call__(self, performance_history):
        metrics = map(self._metric_func, performance_history[self._last_k:])
        return stdev(metrics) / mean(metrics) < self._threshold
