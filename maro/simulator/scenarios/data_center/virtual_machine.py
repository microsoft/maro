# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List


class VirtualMachine:
    """VM object.

    Args:
        id (int): The VM id.
        vcpu_cores_requirement (int): The amount of virtual cores requested by VM.
        memory_requirement (int): The memory requested by VM. The unit is (GBs).
        lifetime (int): The lifetime of the VM, that is, deletion tick - creation tick + 1.
    """
    def __init__(self, id: int, vcpu_cores_requirement: int, memory_requirement: int, lifetime: int):
        # VM Requirement parameters.
        self.id: int = id
        self.vcpu_cores_requirement: int = vcpu_cores_requirement
        self.memory_requirement: int = memory_requirement
        # The VM lifetime which equals to the deletion tick - creation tick + 1.
        self.lifetime: int = lifetime
        # VM utilization list with VM cpu utilization(%) in corresponding tick.
        self._utilization_series: List[float] = []
        # The physical machine Id that the VM is assigned.
        self.pm_id: int = -1
        self.cpu_utilization: float = 0.0
        self.start_tick: int = -1
        self.end_tick: int = -1

    def get_utilization(self, cur_tick: int):
        if cur_tick - self.start_tick > len(self._utilization_series):
            return 0.0

        return self._utilization_series[cur_tick - self.start_tick]

    def add_utilization_series(self, _utilization_series: List[float]):
        """VM CPU utilization list."""
        self._utilization_series = _utilization_series

    @property
    def get_historical_utilization_series(self, cur_tick: int) -> List[float]:
        """"Only expose the CPU utilization series before the current tick."""
        return self._utilization_series[0:cur_tick - self.start_tick]
