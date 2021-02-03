# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

import math
import numpy as np
from pulp import GLPK, LpInteger, LpMaximize, LpProblem, LpStatus, LpVariable, lpSum

from maro.utils import DottableDict

from common import IlpAllocatedVmInfo, IlpFutureVmInfo, IlpPmCapacity, IlpVmInfo


# To indicates not or can not allocate any PM for current VM request.
NOT_ALLOCATE_NOW = -1

class VmSchedulingILP():
    def __init__(self, config: DottableDict, pm_capacity=List[IlpPmCapacity]):
        self._pm_capacity = pm_capacity
        self._pm_num = len(self._pm_capacity)

        # For formulation and action application.
        self.plan_window_size = config.plan_window_size

        # For performance.
        self.core_upper_ratio = 1 - config.performance.core_safety_remaining_ratio
        self.mem_upper_ratio = 1 - config.performance.mem_safety_remaining_ratio

        # For objective.
        self.current_vm_reward_factor = config.objective.current_vm_reward_factor
        self.successful_allocation_decay = config.objective.successful_allocation_decay
        self.allocation_multiple_core_num = config.objective.allocation_multiple_core_num

    def _init_variables(self):
        def _init_with_shape(shape: tuple):
            return np.zeros(shape, dtype=np.int16).tolist()

        # Initialize the PM remaining capacity.
        self._pm_allocated_core = _init_with_shape(shape=(self.plan_window_size, self._pm_num))
        self._pm_allocated_mem = _init_with_shape(shape=(self.plan_window_size, self._pm_num))
        for vm in self._allocated_vm:
            last_tick = min(vm.remaining_lifetime, self.plan_window_size)
            for t in range(last_tick):
                self._pm_allocated_core[t][vm.pm_idx] += vm.core
                self._pm_allocated_mem[t][vm.pm_idx] += vm.mem

        # Initialize the PM-VM mapping variable.
        self._vm_num = len(self._future_vm_req)
        self._mapping = _init_with_shape(shape=(self._pm_num, self._vm_num))
        for pm_idx in range(self._pm_num):
            for vm_idx in range(self._vm_num):
                self._mapping[pm_idx][vm_idx] = LpVariable(
                    name=f"Place_VM{vm_idx}_on_PM{pm_idx}",
                    lowBound=0,
                    upBound=1,
                    cat=LpInteger
                )

    def _add_constraints(self, problem: LpProblem):
        # Mapping limitation: only 1 PM for a VM.
        for vm_idx in range(self._vm_num):
            problem += (
                lpSum(self._mapping[pm_idx][vm_idx] for pm_idx in range(self._pm_num)) <= 1,
                f"Mapping_VM{vm_idx}_to_max_1_PM"
            )

        # PM capacity limitation: core + mem.
        for t in range(self.plan_window_size):
            for pm_idx in range(self._pm_num):
                problem += (
                    lpSum(
                        vm.core * self._mapping[pm_idx][vm_idx]
                        for vm_idx, vm in enumerate(self._future_vm_req)
                        if (vm.arrival_time <= t and vm.remaining_lifetime - vm.arrival_time >= t)
                    ) + self._pm_allocated_core[t][pm_idx] <= self._pm_capacity[pm_idx].core * self.core_upper_ratio,
                    f"T{t}_PM{pm_idx}_core_capacity_limit"
                )
                problem += (
                    lpSum(
                        vm.mem * self._mapping[pm_idx][vm_idx]
                        for vm_idx, vm in enumerate(self._future_vm_req)
                        if (vm.arrival_time <= t and vm.remaining_lifetime - vm.arrival_time >= t)
                    ) + self._pm_allocated_mem[t][pm_idx] <= self._pm_capacity[pm_idx].mem * self.mem_upper_ratio,
                    f"T{t}_PM{pm_idx}_mem_capacity_limit"
                )

    def _set_objective(self, problem: LpProblem):
        # Reward for successful allocation.
        allocation_reward = self.current_vm_reward_factor * lpSum(
            self._mapping[pm_idx][0] for pm_idx in range(self._pm_num)
        )

        if self.allocation_multiple_core_num:
            allocation_reward += lpSum(
                math.pow(self.successful_allocation_decay, self._future_vm_req[vm_idx].arrival_time) * lpSum(
                    self._mapping[pm_idx][vm_idx] for pm_idx in range(self._pm_num)
                ) * self._future_vm_req[vm_idx].core
                for vm_idx in range(1, self._vm_num)
            )
        else:
            allocation_reward += lpSum(
                math.pow(self.successful_allocation_decay, self._future_vm_req[vm_idx].arrival_time) * lpSum(
                    self._mapping[pm_idx][vm_idx] for pm_idx in range(self._pm_num)
                ) for vm_idx in range(1, self._vm_num)
            )

        problem += allocation_reward

    def _formulate_and_solve(self):
        problem = LpProblem(
            name=f"VM_Scheduling_from_tick_{self._env_tick}",
            sense=LpMaximize
        )
        self._init_variables()
        self._add_constraints(problem=problem)
        self._set_objective(problem=problem)
        problem.solve(GLPK(msg=0))

    def choose_pm(
        self,
        env_tick: int,
        vm_req: IlpVmInfo,
        allocated_vm: List[IlpAllocatedVmInfo],
        future_vm_req: List[IlpFutureVmInfo],
    ) -> int:
        self._env_tick = env_tick
        self._allocated_vm = allocated_vm
        self._future_vm_req = [
            IlpFutureVmInfo(
                core=vm_req.core,
                mem=vm_req.mem,
                remaining_lifetime=vm_req.remaining_lifetime,
                arrival_time=0
            )
        ] + future_vm_req

        self._formulate_and_solve()
        for pm_idx in range(self._pm_num):
            if self._mapping[pm_idx][0].varValue:
                return pm_idx

        return NOT_ALLOCATE_NOW
