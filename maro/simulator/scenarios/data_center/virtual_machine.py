# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List


class VirtualMachine:
    def __init__(self, id: str, req_cpu: int, req_mem: int, lifetime: int):
        # Requirement
        self.id: str = id
        self.req_cpu: int = req_cpu
        self.req_mem: int = req_mem
        self.lifetime: int = lifetime
        self._util_series: List[float] = []
        # Utilization
        self.pm_id: int = -1
        self.util_mem: float = 0.0
        self.start_tick: int = -1
        self.end_tick: int = -1

    def add_util_series(self, util_series: List):
        """Load cpu utilization"""

        self._util_series = util_series

    def show_util_series(self):
        return self._util_series
