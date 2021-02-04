# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from statistics import mean

from maro.rl import Scheduler


class SchedulerWithStopping(Scheduler):
    """Linear parameter scheduler with early stopping.

    Args:
        max_ep: Maximum number of episodes.
        warmup_ep: Episode from which early stopping checking is initiated.
        last_k: Number of latest performance records to check for early stopping.
        perf_threshold: The mean of the ``last_k`` performance metric values must be above this value to
            trigger early stopping.
        perf_stability_threshold: The maximum one-step change over the ``last_k`` performance metrics must be
            below this value to trigger early stopping.
    """
    def __init__(self, max_ep, warmup_ep, last_k, perf_threshold, perf_stability_threshold):
        super().__init__(max_ep)
        self._warmup_ep = warmup_ep
        self._last_k = last_k
        self._perf_threshold = perf_threshold
        self._perf_stability_threshold = perf_stability_threshold
        self._performance_history = []

    def check_for_stopping(self):
        if len(self._performance_history) < max(self._last_k, self._warmup_ep):
            return False

        metric_series = list(map(
            lambda record: 1 - record["container_shortage"] / record["order_requirements"],
            self._performance_history[-self._last_k:])
        )
        max_delta = max(abs(me_ - me) / me for me, me_ in zip(metric_series, metric_series[1:]))
        return mean(metric_series) > self._perf_threshold and max_delta < self._perf_stability_threshold

    def record_performance(self, performance):
        self._performance_history.append(performance)
