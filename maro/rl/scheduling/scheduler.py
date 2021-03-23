# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

class Scheduler(object):
    """Scheduler that generates new parameters each iteration.

    Args:
        max_iter (int): Maximum number of iterations. If -1, using the scheduler in a for-loop
            will result in an infinite loop unless the ``check_for_stopping`` method is implemented.
    """

    def __init__(self, max_iter: int = -1):
        if max_iter <= 0 and max_iter != -1:
            raise ValueError("max_iter must be a positive integer or -1.")
        self._max_iter = max_iter
        self._iter_index = -1

    def __iter__(self):
        return self

    def __next__(self):
        self._iter_index += 1
        if self._iter_index == self._max_iter or self.check_for_stopping():
            raise StopIteration

        return self.next_params()

    def next_params(self):
        pass

    def check_for_stopping(self) -> bool:
        return False

    @property
    def iter(self):
        return self._iter_index
