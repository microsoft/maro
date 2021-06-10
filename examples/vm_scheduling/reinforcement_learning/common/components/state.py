# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pdb
import numpy as np
import pandas as pd


PM_ATTRIBUTES = ["cpu_cores_capacity", "memory_capacity", \
                 "cpu_cores_allocated", "memory_allocated"]

VM_ATTRIBUTES = ["cpu_cores_requirement", "memory_requirement", "lifetime", "remain_time", "total_income"]


class VMState(object):
    def __init__(
        self, pm_num, durations, vm_state_path, vm_window_size, pm_window_size
    ):
        self._pm_num = pm_num
        self._durations = durations
        self._vm_states = np.load(vm_state_path)
        self._dim = (pm_num * 2) * pm_window_size + len(VM_ATTRIBUTES) * vm_window_size

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
            env.frame_index::PM_ATTRIBUTES
        ].reshape(self._pm_num, len(PM_ATTRIBUTES))

        # normalize the attributes of pms' cpu and memory
        self._max_cpu_capacity = np.max(total_pm_info[:, 0])
        self._max_memory_capacity = np.max(total_pm_info[:, 1])
        total_pm_info[:, 2] /= self._max_cpu_capacity
        total_pm_info[:, 3] /= self._max_memory_capacity

        # get the remaining cpu and memory of the pms
        remain_cpu = (1 - total_pm_info[:, 2]).reshape(1, self._pm_num, 1)
        remain_memory = (1 - total_pm_info[:, 3]).reshape(1, self._pm_num, 1)

        # get the pms' information
        total_pm_info = np.concatenate((remain_cpu, remain_memory), axis=2)

        # get the sequence pms' information
        self._history_pm_state = np.concatenate((self._history_pm_state, total_pm_info), axis=0)
        sequence_pm_info = self._history_pm_state[-self._pm_window_size:, :, :].copy()

        return sequence_pm_info

    def _get_vm_state(self, env, decision_event):
        if self._vm_window_size == 1:
            # get the vm's infomation
            vm_info = np.array([
                decision_event.vm_cpu_cores_requirement,
                decision_event.vm_memory_requirement,
                min(self._durations - env.tick, decision_event.vm_lifetime) / 200,
                (self._durations - env.tick) * 1.0 / 200,
                decision_event.vm_unit_price * min(self._durations - env.tick, decision_event.vm_lifetime)
            ], dtype=np.float)
            vm_info[0] /= self._max_cpu_capacity
            vm_info[1] /= self._max_memory_capacity

            return vm_info
        else:
            # get the sequence vms' information
            total_vm_info = np.zeros((self._vm_window_size, len(VM_ATTRIBUTES)))

            for idx in range(self._st, self._st + self._vm_window_size):
                if idx < self._vm_states.shape[0]:
                    vm_info = self._vm_states[idx].copy()
                    vm_info[0] /= self._max_cpu_capacity
                    vm_info[1] /= self._max_memory_capacity
                    vm_info[4] = vm_info[4] * min(self._durations - vm_info[3], vm_info[2])
                    vm_info[2] = (vm_info[2] * 1.0) / 200
                    vm_info[3] = (self._durations - vm_info[3]) * 1.0 / 200
                else:
                    vm_info = np.zeros(len(VM_ATTRIBUTES), dtype=np.float)

                total_vm_info[self._vm_window_size - (idx - self._st + 1), :] = vm_info

            self._st = (self._st + 1) % self._vm_states.shape[0]

            return total_vm_info

    def _get_legal_pm(self, decision_event, total_pm_info):
        # get the legal pm
        legal_pm = np.zeros(self._pm_num + 1)
        if len(decision_event.valid_pms) <= 0:
            # no pm is available
            legal_pm[self._pm_num] = 1
        else:
            legal_pm[self._pm_num] = 1

            remain_cpu_dict = dict()
            for pm in decision_event.valid_pms:
                # if two pm has same remaining cpu, only choose the one which has smaller id
                if total_pm_info[-1, pm, 0] not in remain_cpu_dict.keys():
                    remain_cpu_dict[total_pm_info[-1, pm, 0]] = 1
                    legal_pm[pm] = 1
                else:
                    legal_pm[pm] = 0

        return legal_pm

    @property
    def dim(self):
        return self._dim

    @property
    def pm_num(self):
        return self._pm_num
