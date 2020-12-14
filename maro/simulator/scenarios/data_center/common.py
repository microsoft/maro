# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum
from typing import List

from .virtual_machine import VirtualMachine


class Action:
    """Data center scenario action object, which was used to pass action from agent to business engine.

    Args:
        vm_id (int): The VM id.
    """
    def __init__(self, vm_id: int):
        self.vm_id = vm_id


class PostponeAction(Action):
    """Postpone action object.

    Args:
        vm_id (int): The VM id.
        postpone_frequency (int): The number of times be postponed.
    """
    def __init__(self, vm_id: int, postpone_frequency: int):
        super().__init__(vm_id)
        self.postpone_frequency = postpone_frequency


class AssignAction(Action):
    """Assign action object.

    Args:
        vm_id (int): The VM id.
        pm_id (int): The physical machine id assigned to the VM.
    """
    def __init__(self, vm_id: int, pm_id: int):
        super().__init__(vm_id)
        self.pm_id = pm_id


class VmRequestPayload:
    """Payload for the VM requirement.

    Args:
        vm_info (VirtualMachine): The VM information.
        remaining_buffer_time (int): The remaining buffer time.
    """
    summary_key = ["vm_info", "remaining_buffer_time"]

    def __init__(self, vm_info: VirtualMachine, remaining_buffer_time: int):
        self.vm_info = vm_info
        self.remaining_buffer_time = remaining_buffer_time


class VmFinishedPayload:
    """Payload for the VM finished.

    Args:
        vm_id (int): The id of the VM.
    """
    summary_key = ["vm_id"]

    def __init__(self, vm_id: int):
        self.vm_id = vm_id


class DecisionPayload:
    """Decision event in Data center scenario that contains information for agent to choose action.

    Args:
        valid_pms (List[]): A list contains ValidPhysicalMachine object.
        vm_id (int): The id of the VM.
        vm_cpu_cores_requirement (int): The CPU requested by VM.
        vm_memory_requirement (int): The memory requested by VM.
        remaining_buffer_time (int): The remaining buffer time.
    """
    summary_key = ["valid_pms", "vm_id", "vm_cpu_cores_requirement", "vm_memory_requirement", "remaining_buffer_time"]

    def __init__(
        self,
        valid_pms: List,
        vm_id: int,
        vm_cpu_cores_requirement: int,
        vm_memory_requirement: int,
        remaining_buffer_time: int
    ):
        self.valid_pms = valid_pms
        self.vm_id = vm_id
        self.vm_cpu_cores_requirement = vm_cpu_cores_requirement
        self.vm_memory_requirement = vm_memory_requirement
        self.remaining_buffer_time = remaining_buffer_time


class ValidPhysicalMachine:
    """The object for the valid PM which will be sent to the agent."""
    def __init__(self, pm_id: int, remaining_cpu: int, remaining_mem: int):
        self.pm_id = pm_id
        self.remaining_cpu = remaining_cpu
        self.remaining_mem = remaining_mem


class PostponeType(Enum):
    """Postpone type."""
    # Postpone the VM requirement due to the resource exhaustion.
    Resource = 'resource'
    # Postpone the VM requirement due to the agent's decision.
    Agent = 'agent'


class Latency:
    """Accumulative latency.

    Two types of the latency.
    1. The accumulative latency triggered by the algorithm inaccurate predictions.
    2. The accumulative latency triggered by the resource exhaustion.
    """
    def __init__(self):
        self.due_to_agent: int = 0
        self.due_to_resource: int = 0

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'Latency(Agent={self.due_to_agent}, Resource={self.due_to_resource})'
