# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List


class PhysicalMachine:
    def __init__(self, id: int, cap_cpu: int, cap_mem: int):
        self.id = id
        self.cap_cpu: int = cap_cpu
        self.cap_mem: int = cap_mem
        self.vm_list: List[int] = []
        self.req_cpu: int = -1
        self.req_mem: int = -1
        self.util_cpu: int = -1
        self.util_mem: int = -1
        self.util_series: List = []

    def add_vm(self, vm_id: int):
        self.vm_list.append(vm_id)
