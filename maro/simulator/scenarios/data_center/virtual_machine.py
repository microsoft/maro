# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List


class VirtualMachine:
    """VM object.

    Args:
        id (int): VM id loaded from public data set.
        req_cpu (int): The amount of virtual core requested by VM.
        req_mem (int): The memory requested by VM.
        lifetime (int): The lifetime of VM, that is, deletion tick - creation tick.
    """
    def __init__(self, id: str, req_cpu: int, req_mem: int, lifetime: int):
        # VM Requirement parameters.
        self.id: str = id
        self.req_cpu: int = req_cpu
        self.req_mem: int = req_mem
        # The VM lifetime which equals to the deletion tick - creation tick.
        self.lifetime: int = lifetime
        # VM utilization list with VM cpu utilization(%) in corresponding tick.
        self._util_series: List[float] = []
        # The physical machine Id that the VM is assigned.
        self.pm_id: int = -1
        self.util_cpu: float = 0.0
        self.start_tick: int = -1
        self.end_tick: int = -1

    def get_util(self, cur_tick: int):
        return self._util_series[cur_tick - self.start_tick]

    def add_util_series(self, util_series: List[float]):
        """VM CPU utilization list."""
        self._util_series = util_series

    @property
    def get_historical_util_series(self, cur_tick: int) -> List[float]:
        """"Only expose the CPU utilization series before the current tick."""
        return self._util_series[0:cur_tick - self.start_tick]
