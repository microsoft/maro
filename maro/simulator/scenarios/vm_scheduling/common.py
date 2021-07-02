# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

from .virtual_machine import VirtualMachine


class Action:
    """VM Scheduling scenario action object, which was used to pass action from agent to business engine.

    Args:
        vm_id (int): The VM id.
    """

    def __init__(self, vm_id: int):
        self.vm_id = vm_id

    def __repr__(self):
        return "%s {vm_id: %r}" % (self.__class__.__name__, self.vm_id)


class PostponeAction(Action):
    """Postpone action object.

    Args:
        vm_id (int): The VM id.
        postpone_step (int): The number of times be postponed.
    """

    def __init__(self, vm_id: int, postpone_step: int):
        super().__init__(vm_id)
        self.postpone_step = postpone_step

    def __repr__(self):
        return "%s {vm_id: %r, postpone_step: %r}" % (self.__class__.__name__, self.vm_id, self.postpone_step)


class AllocateAction(Action):
    """Allocation action object.

    Args:
        vm_id (int): The VM id.
        pm_id (int): The id of the physical machine where the VM will be allocated.
    """

    def __init__(self, vm_id: int, pm_id: int):
        super().__init__(vm_id)
        self.pm_id = pm_id

    def __repr__(self):
        return "%s {vm_id: %r, pm_id: %r}" % (self.__class__.__name__, self.vm_id, self.pm_id)


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

    def __repr__(self):
        return "%s {vm_info: %r, remaining_buffer_time: %r}" % \
            (self.__class__.__name__, self.vm_info, self.remaining_buffer_time)


class DecisionPayload:
    """Decision event in VM Scheduling scenario that contains information for agent to choose action.

    Args:
        frame_index (int): The current frame index (converted by tick).
        valid_pms (List[int]): A list contains pm id of all valid pms.
        vm_id (int): The id of the VM.
        vm_cpu_cores_requirement (int): The CPU requested by VM.
        vm_memory_requirement (int): The memory requested by VM.
        remaining_buffer_time (int): The remaining buffer time.
    """
    summary_key = [
        "frame_index", "valid_pms", "vm_id", "vm_cpu_cores_requirement", "vm_memory_requirement",
        "remaining_buffer_time"
    ]

    def __init__(
        self,
        frame_index: int,
        valid_pms: List[int],
        vm_id: int,
        vm_cpu_cores_requirement: int,
        vm_memory_requirement: int,
        vm_sub_id: int,
        vm_category: int,
        remaining_buffer_time: int
    ):
        self.frame_index = frame_index
        self.valid_pms = valid_pms
        self.vm_id = vm_id
        self.vm_cpu_cores_requirement = vm_cpu_cores_requirement
        self.vm_memory_requirement = vm_memory_requirement
        self.vm_sub_id = vm_sub_id
        self.vm_category = vm_category
        self.remaining_buffer_time = remaining_buffer_time

    def __repr__(self):
        return "%s {frame_index: %r, valid_pms: %r, vm_id: %r, vm_cpu_cores_requirement: %r, \
            vm_memory_requirement: %r, vm_sub_id: %r, vm_category: %r, remaining_buffer_time: %r}" % \
            (
                self.__class__.__name__,
                self.frame_index,
                self.valid_pms,
                self.vm_id,
                self.vm_cpu_cores_requirement,
                self.vm_memory_requirement,
                self.vm_sub_id,
                self.vm_category,
                self.remaining_buffer_time
            )


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
        return "%s {due_to_agent: %r, due_to_resource: %r}" % \
            (self.__class__.__name__, self.due_to_agent, self.due_to_resource)
