# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

from maro.rl import Shaper

TICK_ATTRIBUTES = ["current_tick"]
PM_ATTRIBUTES = ["cpu_cores_capacity", "memory_capacity", \
                 "cpu_cores_allocated", "memory_allocated"]
VM_ATTRIBUTES = ["cpu_cores_requirement", "memory_requirement", "lifetime"]
# VM_ATTRIBUTES = ["cpu_cores_requirement", "memory_requirement", "sub_id", "category"]

class VMStateShaper(Shaper):
    def __init__(self, pm_num):
        super().__init__()
        self._pm_num = pm_num
        self._dim = pm_num * (len(PM_ATTRIBUTES) + 1) + len(VM_ATTRIBUTES) + 1

    def __call__(self, decision_event, env):
        total_pm_info = env.snapshot_list["pms"][
            env.frame_index::PM_ATTRIBUTES
        ].reshape(self._pm_num, len(PM_ATTRIBUTES))
        max_cpu_capacity = np.max(total_pm_info[:, 0])
        max_memory_capacity = np.max(total_pm_info[:, 1])
        total_pm_info[:, 2] /= max_cpu_capacity
        total_pm_info[:, 3] /= max_memory_capacity
        total_pm_info[:, 0] /= max_cpu_capacity
        total_pm_info[:, 1] /= max_memory_capacity
        pm_id = np.arange(self._pm_num).reshape(self._pm_num, 1)
        total_pm_info = np.concatenate((pm_id, total_pm_info), axis=1)
        """
        vm_info = np.array([
            decision_event.vm_cpu_cores_requirement, decision_event.vm_memory_requirement,
            decision_event.vm_sub_id, decision_event.vm_category, decision_event.vm_lifetime
        ])
        """
        vm_info = np.array([
            decision_event.vm_cpu_cores_requirement, 
            decision_event.vm_memory_requirement, 
            decision_event.vm_lifetime
        ])
        vm_info[0] /= max_cpu_capacity
        vm_info[1] /= max_memory_capacity

        state = np.concatenate((total_pm_info.flatten(), vm_info))
        state = np.concatenate((np.array([env.tick]), state))

        # print(state)
        
        legal_pm = np.zeros(self._pm_num + 1)
        
        """
        if decision_event.vm_cpu_cores_requirement == 2:
            legal_pm[0] = 1
        else:
            if len(decision_event.valid_pms) <= 0:
                legal_pm[self._pm_num] = 1
            else:
                for pm in decision_event.valid_pms:
                    legal_pm[pm] = 1
        """
        
        if len(decision_event.valid_pms) <= 0:
                legal_pm[self._pm_num] = 1
        else:
            for pm in decision_event.valid_pms:
                if total_pm_info[pm, 3] == 0:
                    legal_pm[pm] = 1
                    break
                else:
                    legal_pm[pm] = 1
        
        return state, legal_pm

    @property
    def dim(self):
        return self._dim
