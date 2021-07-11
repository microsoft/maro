# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import pandas as pd


PM_EXTRACTED_ATTRIBUTES = [
    "cpu_cores_capacity", "memory_capacity", \
    "cpu_cores_allocated", "memory_allocated"
]
# use the remaining cpu and memory of the pms
PM_ATTRIBUTES = ["cpu_cores_remaining", "memory_remaining"]
# use the require cpu and memory of the vm, if peeking, some extra information will also be used
VM_ATTRIBUTES = ["cpu_cores_requirement", "memory_requirement", "lifetime", "remain_time", "total_income"]


class VMState(object):
    def __init__(
        self, pm_num, durations, normalize_durations, 
        peeking, vm_state_path, vm_window_size, pm_window_size
    ):
        self._pm_num = pm_num
        self._durations = durations
        self._normalize_durations = normalize_durations
        self._vm_states = np.load(vm_state_path)
        self._peeking = peeking

        pm_dim = len(PM_ATTRIBUTES)        
        vm_dim = len(VM_ATTRIBUTES) if self._peeking else 2
        self._dim = (pm_num * pm_dim) * pm_window_size + vm_dim * vm_window_size

        self._history_pm_state = np.zeros((pm_window_size - 1, self._pm_num, 2))
        self._st, self._vm_window_size, self._pm_window_size = 0, vm_window_size, pm_window_size

    def __call__(self, env, decision_event):
        # get state
        pm_state = self._get_pm_state(env)
        vm_state = self._get_vm_state(env, decision_event)
        state = np.concatenate((pm_state.flatten(), vm_state.flatten()))
        # get legal pm
        legal_pm = self._get_legal_pm(decision_event, pm_state)

        return state, legal_pm

    def _get_pm_state(self, env):
        total_pm_info = env.snapshot_list["pms"][
            env.frame_index::PM_EXTRACTED_ATTRIBUTES
        ].reshape(self._pm_num, len(PM_EXTRACTED_ATTRIBUTES))

        # normalize the attributes of pms' cpu and memory
        self._max_cpu_capacity = np.max(total_pm_info[:, 0])
        self._max_memory_capacity = np.max(total_pm_info[:, 1])

        # get the remaining cpu and memory of the pms
        remain_cpu = (total_pm_info[:, 0] - total_pm_info[:, 2]) / self._max_cpu_capacity
        remain_cpu = remain_cpu.reshape(1, self._pm_num, 1)
        remain_memory = (total_pm_info[:, 1] - total_pm_info[:, 3]) / self._max_memory_capacity
        remain_memory = remain_memory.reshape(1, self._pm_num, 1)

        # get the pms' information
        total_pm_info = np.concatenate((remain_cpu, remain_memory), axis=2)

        # get the sequence pms' information
        self._history_pm_state = np.concatenate((self._history_pm_state, total_pm_info), axis=0)
        sequence_pm_info = self._history_pm_state[-self._pm_window_size:, :, :].copy()

        return sequence_pm_info

    def _get_vm_state(self, env, decision_event):
        if self._vm_window_size == 1:
            # get the vm's infomation
            vm_info = [
                decision_event.vm_cpu_cores_requirement,
                decision_event.vm_memory_requirement,
            ]
            vm_info[0] /= self._max_cpu_capacity
            vm_info[1] /= self._max_memory_capacity
            if self._peeking:
                vm_info.extend(
                    [min(self._durations - env.tick, decision_event.vm_lifetime) / self._normalize_durations,
                    (self._durations - env.tick) * 1.0 / self._normalize_durations,
                    decision_event.vm_unit_price * min(self._durations - env.tick, decision_event.vm_lifetime)]
                )
            vm_info = np.array(vm_info)
                
            return vm_info
        else:
            if self._vm_states[self._st][0] != decision_event.vm_cpu_cores_requirement \
                or self._vm_states[self._st][1] != decision_event.vm_memory_requirement:
                raise ValueError(
                    f"the vm states and current simulation must match!"
                )

            # get the sequence vms' information
            if self._peeking:
                total_vm_info = np.zeros((self._vm_window_size, len(VM_ATTRIBUTES)))
            else:
                total_vm_info = np.zeros((self._vm_window_size, 2))

            for idx in range(self._st, self._st + self._vm_window_size):
                if idx < self._vm_states.shape[0]:
                    vm_info = self._vm_states[idx].copy()
                    vm_info[0] /= self._max_cpu_capacity
                    vm_info[1] /= self._max_memory_capacity
                    if self._peeking:
                        vm_info[4] = vm_info[4] * min(self._durations - vm_info[3], vm_info[2])
                        vm_info[2] = (vm_info[2] * 1.0) / self._normalize_durations
                        vm_info[3] = (self._durations - vm_info[3]) * 1.0 / self._normalize_durations

                total_vm_info[self._vm_window_size - (idx - self._st + 1), :] = vm_info

            self._st = (self._st + 1) % self._vm_states.shape[0]

            return total_vm_info

    def _get_legal_pm(self, decision_event, total_pm_info):
        # get the legal pm
        legal_pm = np.zeros(self._pm_num + 1)
        legal_pm[self.pm_num] = 1
        if len(decision_event.valid_pms) > 0:
            remain_cpu_dict = dict()
            for pm in decision_event.valid_pms:
                # if two pm has same remaining cpu, only choose the one which has smaller id
                if total_pm_info[-1, pm, 0] not in remain_cpu_dict.keys():
                    remain_cpu_dict[total_pm_info[-1, pm, 0]] = 1
                    legal_pm[pm] = 1

        return legal_pm

    @property
    def dim(self):
        return self._dim

    @property
    def pm_num(self):
        return self._pm_num
