# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Callable

from statistics import mean, stdev

from .abs_early_stopping_checker import AbsEarlyStoppingChecker


class SimpleEarlyStoppingChecker(AbsEarlyStoppingChecker):
    def __init__(self, last_k: int, performance_metric_func: Callable, threshold: float):
        super().__init__()
        self._last_k = last_k
        self._performance_metric_func = performance_metric_func
        self._threshold = threshold

    def __call__(self, performance_history):
        metrics = map(self._performance_metric_func, performance_history[self._last_k:])
        return stdev(metrics) / mean(metrics) < self._threshold
