# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

from .virtual_machine import VirtualMachine


class Action:
    """Data center scenario action object, which was used to pass action from agent to business engine.

    Args:
        assign (bool): Assign the VM to the PM or not.
        vm_id (int): The VM id.
        pm_id (int): The physical machine id assigned to this vm.
        remaining_buffer_time (int): The remaining buffer time to assign this VM.
    """

    def __init__(self, assign: bool, vm_id: int, pm_id: int, remaining_buffer_time: int):
        self.assign = assign
        self.vm_id = vm_id
        self.pm_id = pm_id
        self.remaining_buffer_time = remaining_buffer_time


class VmRequirementPayload:
    """Payload for the VM requirement.

    Args:
        vm_info (VirtualMachine): The VM information.
        remaining_buffer_time (int): The remaining buffer time.
    """

    summary_key = ["vm_info", "remaining_buffer_time"]

    def __init__(self, vm_req: VirtualMachine, remaining_buffer_time: int):
        self.vm_req = vm_req
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
        valid_pm (List[dict]): A list contains dictionaries with 'pm_id', 'cpu', 'mem'.
        vm_id (int): The id of the VM.
        vm_req_cpu (int): The CPU requested by VM.
        vm_req_mem (int): The memory requested by VM.
        remaining_buffer_time (int): The remaining buffer time.
    """

    summary_key = ["valid_pm", "vm_id", "vm_req_cpu", "vm_req_mem", "remaining_buffer_time"]

    def __init__(self, valid_pm: List[dict], vm_id: int, vm_req_cpu: int, vm_req_mem: int, remaining_buffer_time: int):
        self.valid_pm = valid_pm
        self.vm_id = vm_id
        self.vm_req_cpu = vm_req_cpu
        self.vm_req_mem = vm_req_mem
        self.remaining_buffer_time = remaining_buffer_time


class ValidPm:
    """The object for the valid PM which will be sent to the agent."""

    def __init__(self, pm_id: int, remaining_cpu: int, remaining_mem: int):
        self.pm_id = pm_id
        self.remaining_cpu = remaining_cpu
        self.remaining_mem = remaining_mem


class Latency:
    """Accumulative latency.

    Two types of the latency.
    1. The accumulative latency triggered by the algorithm inaccurate predictions.
    2. The accumulative latency triggered by the resource exhaustion.
    """
    def __init__(self):
        self.latency_due_to_agent: int = 0
        self.latency_due_to_resource: int = 0
