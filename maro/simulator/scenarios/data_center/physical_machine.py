# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Set


class PhysicalMachine:
    """Physical machine object.

    Args:
        id (int): PM id, from 0 to N. N means the amount of PM, which can be set in config.
        cpu_cores_capacity (int): The capacity of cores of the PM, which can be set in config.
        memory_capacity (int): The capacity of memory of the PM, which can be set in config.
    """
    def __init__(self, id: int, cpu_cores_capacity: int, memory_capacity: int):
        # Required parameters.
        self.id: int = id
        self.cpu_cores_capacity: int = cpu_cores_capacity
        self.memory_capacity: int = memory_capacity
        # PM resource.
        self._live_vms: Set(int) = set()
        self.cpu_allocation: int = 0
        self.memory_allocation: int = 0
        self._cpu_utilization: float = 0.0
        self._cpu_utilization_series: List[float] = []
        # Energy consumption converted by cpu utilization.
        self._energy_consumption: List[float] = []

    @property
    def live_vms(self) -> Set(int):
        return self._live_vms

    def place_vm(self, vm_id: int):
        self._live_vms.add(vm_id)

    def remove_vm(self, vm_id: int):
        self._live_vms.remove(vm_id)

    @property
    def cpu_utilization(self) -> float:
        # PM CPU utilization (%).
        return self._cpu_utilization

    def update_utilization(self, tick: int, cpu_utilization: float):
        if tick > len(self._cpu_utilization_series):
            raise Exception(f"The tick: '{tick}' is invalid.")

        # Update CPU utilization.
        self._cpu_utilization = cpu_utilization

        # Update the utilization series.
        if tick == len(self._cpu_utilization_series):
            self._cpu_utilization_series.append(cpu_utilization)
        elif tick < len(self._cpu_utilization_series):
            self._cpu_utilization_series[tick] = cpu_utilization

    def update_energy(self, tick: int, cur_energy: float):
        if tick > len(self._energy_consumption):
            raise Exception(f"The tick: '{tick}' is invalid.")

        # Update the energy series.
        if tick == len(self._energy_consumption):
            self._energy_consumption.append(cur_energy)
        elif tick < len(self._energy_consumption):
            self._energy_consumption[tick] = cur_energy
