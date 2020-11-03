# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

class VirtualMachine:
    def __init__(self, id: int, req_cpu: int, req_mem: int):
        self.id: int = id
        self.req_cpu: int = req_cpu
        self.req_mem: int = req_mem
        self.pm_id: int = -1
        self.util_cpu: int = 0
        self.util_mem: int = 0
        self.start_tick: int = -1
        self.end_tick: int = -1
        self.util_series: List = []
        