# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod


class AbsExplorer(ABC):
    """Abstract explorer class for generating exploration rates.

    """
    def __init__(self):
        pass

    # TODO: performance: summary -> total perf (current version), details -> per-agent perf
    @abstractmethod
    def generate_epsilon(self, current_ep: int, max_ep: int, performance_history=None):
        """Generate an exploration rate based on the performance history.
        """
        return NotImplemented
