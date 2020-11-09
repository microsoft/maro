# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .virtual_machine import VirtualMachine


class Action:
    """Data center scenario action object, which was used to pass action from agent to business engine.

    Args:
        vm (VirtualMachine): The virtual machine object, which only contains id, requirement resource, lifetime.
        pm_id (int): The physical machine id assigned to this vm.
        buffertime (int): The buffer time to assign this VM.
    """

    def __init__(self, assign: bool, vm_req: VirtualMachine, buffer_time: int, pm_id: int = None):
        self.assign = assign
        self.vm_req = vm_req
        self.pm_id = pm_id
        self.buffer_time = buffer_time


class Payload:
    """Payload for the VM information.

    Args:
        vm_info (VirtualMachine): The VM information
        buffer time (int): The remaining buffer time
    """

    summary_key = ["vm_info", "buffer_time"]

    def __init__(self, vm_info: VirtualMachine, buffer_time: int):
        self.vm_info = vm_info
        self.buffer_time = buffer_time


class Latency:
    """Buffer time.

    1. Triggered by the resource exhaustion (resource_buffer_time).
    2. Triggered by the algorithm inaccurate predictions (algorithm_buffer_time).
    """
    def __init__(self):
        self.algorithm_buffer_time = 0
        self.resource_buffer_time = 0
