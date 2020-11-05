# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List


class VirtualMachine:
    def __init__(self, id: int, req_cpu: int, req_mem: int, lifetime: int):
        # Requirement
        self.id: int = id
        self.req_cpu: int = req_cpu
        self.req_mem: int = req_mem
        self.lifetime: int = lifetime
        self.util_series: List = []
        # Utilization
        self.pm_id: int = -1
        self.util_cpu: int = 0
        self.util_mem: int = 0
        self.start_tick: int = -1
        self.end_tick: int = -1

    def add_util_series(self):
        """Load cpu utilization
        """

        self.util_series = None
