# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Set

from maro.backends.frame import NodeAttribute, NodeBase, node

from .virtual_machine import VirtualMachine


@node("pms")
class PhysicalMachine(NodeBase):
    """Physical machine node definition in frame."""
    # Initial parameters.
    id = NodeAttribute("i")
    cpu_cores_capacity = NodeAttribute("i2")
    memory_capacity = NodeAttribute("i2")
    # Statistical features.
    cpu_cores_allocated = NodeAttribute("i2")
    memory_allocated = NodeAttribute("i2")

    cpu_utilization = NodeAttribute("f")
    energy_consumption = NodeAttribute("f")

    # PM type: empty: -1, non-oversubscribable is 0; oversubscribable is 1.
    oversubscribable = NodeAttribute("i2")

    def __init__(self):
        """Internal use for reset."""
        self._id = 0
        self._init_cpu_cores_capacity = 0
        self._init_memory_capacity = 0
        self._init_type = 0
        # PM resource.
        self._live_vms: Set[int] = set()

    def update_cpu_utilization(self, vm: VirtualMachine = None, cpu_utilization: float = None):
        if vm is None and cpu_utilization is None:
            raise Exception(f"Wrong calling method {self.update_cpu_utilization.__name__}")

        if vm is not None:
            cpu_utilization = (
                (self.cpu_cores_capacity * self.cpu_utilization + vm.cpu_cores_requirement * vm.cpu_utilization)
                / self.cpu_cores_capacity
            )

        self.cpu_utilization = round(max(0, cpu_utilization), 2)

    def set_init_state(self, id: int, cpu_cores_capacity: int, memory_capacity: int, oversubscribable: int):
        """Set initialize state, that will be used after frame reset.

        Args:
            id (int): PM id, from 0 to N. N means the amount of PM, which can be set in config.
            cpu_cores_capacity (int): The capacity of cores of the PM, which can be set in config.
            memory_capacity (int): The capacity of memory of the PM, which can be set in config.
            oversubscribable (int): The type of the PM. 0 means empty, -1 means non-oversubscribable; 1 means oversubscribable.
        """
        self._id = id
        self._init_cpu_cores_capacity = cpu_cores_capacity
        self._init_memory_capacity = memory_capacity
        self._init_type = oversubscribable

        self.reset()

    def reset(self):
        """Reset to default value."""
        # When we reset frame, all the value will be set to 0, so we need these lines.
        self.id = self._id
        self.cpu_cores_capacity = self._init_cpu_cores_capacity
        self.memory_capacity = self._init_memory_capacity
        self.oversubscribable = self._init_type

        self._live_vms.clear()

        self.cpu_cores_allocated = 0
        self.memory_allocated = 0

        self.cpu_utilization = 0.0
        self.energy_consumption = 0.0

    @property
    def live_vms(self) -> Set[int]:
        return self._live_vms

    def allocate_vms(self, vm_ids: List[int]):
        for vm_id in vm_ids:
            self._live_vms.add(vm_id)

    def deallocate_vms(self, vm_ids: List[int]):
        for vm_id in vm_ids:
            self._live_vms.remove(vm_id)
