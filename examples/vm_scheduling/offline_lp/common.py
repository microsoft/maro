# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

class IlpPmCapacity():
    def __init__(self, core: int, mem: int):
        self.core = core
        self.mem = mem

class IlpVmInfo():
    def __init__(self, core: int, mem: int, remaining_lifetime: int=-1):
        self.core: int = core
        self.mem: int = mem
        self.remaining_lifetime: int = remaining_lifetime

class IlpAllocatedVmInfo(IlpVmInfo):
    def __init__(self, pm_idx: int, core: int, mem: int, remaining_lifetime: int=-1):
        super().__init__(core, mem, remaining_lifetime)
        self.pm_idx = pm_idx

class IlpFutureVmInfo(IlpVmInfo):
    def __init__(self, core: int, mem: int, remaining_lifetime: int, arrival_time: int):
        super().__init__(core, mem, remaining_lifetime)
        self.arrival_time = arrival_time

    def __repr__(self):
        return (
            f"core: {self.core}, "
            f"mem: {self.mem}, "
            f"remaining_lifetime: {self.remaining_lifetime}, "
            f"arrival_time: {self.arrival_time}"
        )
