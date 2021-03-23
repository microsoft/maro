# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

from maro.rl import Shaper

PM_ATTRIBUTES = ["cpu_cores_capacity", "memory_capacity", "pm_type", \
                 "cpu_cores_allocated", "memory_allocated", "cpu_utilization", "energy_consumption"]
VM_ATTRIBUTES = ["cpu_cores_requirement", "memory_requirement", "sub_id", "category", "lifetime"]


class VMStateShaper(Shaper):
    def __init__(self, pm_num):
        super().__init__()
        self._pm_num = pm_num
        self._dim = pm_num * len(PM_ATTRIBUTES) + len(VM_ATTRIBUTES)

    def __call__(self, decision_event, env):
        total_pm_info = env.snapshot_list["pms"][
            env.frame_index::PM_ATTRIBUTES
        ].reshape(self._pm_num, len(PM_ATTRIBUTES))
        vm_info = np.array([
            decision_event.vm_cpu_cores_requirement, decision_event.vm_memory_requirement,
            decision_event.vm_sub_id, decision_event.vm_category, decision_event.vm_lifetime
        ])
        state = np.concatenate((total_pm_info.flatten(), vm_info))
        legal_pm = np.zeros(self._pm_num + 1)
        if len(decision_event.valid_pms) <= 0:
            legal_pm[self._pm_num] = 1
        else:
            for pm in decision_event.valid_pms:
                legal_pm[pm] = 1
        return state, legal_pm

    @property
    def dim(self):
        return self._dim
