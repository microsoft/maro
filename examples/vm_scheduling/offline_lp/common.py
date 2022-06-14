# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass


@dataclass
class IlpPmCapacity:
    core: int
    mem: int


@dataclass
class IlpVmInfo:
    id: int = -1
    pm_idx: int = -2
    core: int = -1
    mem: int = -1
    lifetime: int = -1
    arrival_env_tick: int = -1

    def remaining_lifetime(self, env_tick: int):
        return self.lifetime - (env_tick - self.arrival_env_tick)
