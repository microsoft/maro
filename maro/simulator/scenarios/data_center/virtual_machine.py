# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List


class VirtualMachine:
    """VM object.

    Args:
        id (int): VM id loaded from public data set.
        req_cpu (int): The amount of virtual core requested by VM.
        rea_mem (int): The memory requested by VM.
        lifetime (int): The lifetime of VM, that is, deletion time - creation time.
    """
    def __init__(self, id: str, req_cpu: int, req_mem: int, lifetime: int):
        # VM Requirement parameters.
        self.id: str = id
        self.req_cpu: int = req_cpu
        self.req_mem: int = req_mem
        self.lifetime: int = lifetime
        self._util_series: List[float] = []
        # Utilization
        self.pm_id: int = -1
        self.util_mem: float = 0.0
        self.start_tick: int = -1
        self.end_tick: int = -1

    def add_util_series(self, util_series: List):
        """Load cpu utilization."""
        self._util_series = util_series

    @property
    def util_series(self) -> List[float]:
        return self._util_series
