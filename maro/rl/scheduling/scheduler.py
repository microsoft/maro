# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

class Scheduler(object):
    """Scheduler that generates new parameters each iteration.

    Args:
        max_iter (int): Maximum number of iterations. If -1, the next() method can be called
            an unlimited number of times 
    """

    def __init__(self, max_iter: int = -1):
        if max_iter <= 0 and max_iter != -1:
            raise ValueError("max_iter must be a positive integer or -1.")
        self._max_iter = max_iter
        self._current_iter = -1
        self.performance_history = []

    def __iter__(self):
        return self

    def __next__(self):
        self._current_iter += 1
        if self._current_iter == self._max_iter or self.check_for_stopping():
            raise StopIteration

        return self.next_params()

    def next_params(self):
        pass

    def check_for_stopping(self) -> bool:
        return False

    @property
    def current_iter(self):
        return self._current_iter

    def record_performance(self, performance):
        self.performance_history.append(performance)
