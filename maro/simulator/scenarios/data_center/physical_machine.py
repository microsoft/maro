# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Set


class PhysicalMachine:
    def __init__(self, id: int, cap_cpu: int, cap_mem: int):
        # defualt
        self.id = id
        self.cap_cpu: int = cap_cpu
        self.cap_mem: int = cap_mem
        # PM resource
        self._vm_set: Set(int) = set()
        self.req_cpu: int = -1
        self.req_mem: int = -1
        self.util_mem: int = -1
        self._util_series: List[int] = []

    def add_vm(self, vm_id: int):
        self._vm_set.add(vm_id)

    def remove_vm(self, vm_id: int):
        self._vm_set.remove(vm_id)

    def show_vm_set(self):
        return self._vm_set

    def update_util_series(self, cur_util_mem: int):
        self._util_series.append(cur_util_mem)
