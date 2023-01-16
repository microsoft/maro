# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import os
import timeit
from typing import List

import numpy as np
from common import IlpPmCapacity, IlpVmInfo
from pulp import GLPK, PULP_CBC_CMD, LpInteger, LpMaximize, LpProblem, LpStatus, LpVariable, lpSum

from maro.utils import DottableDict, Logger

# To indicate the decision of not allocate or cannot allocate any PM for current VM request.
NOT_ALLOCATE_NOW = -1


class VmSchedulingILP:
    def __init__(self, config: DottableDict, pm_capacity: List[IlpPmCapacity], logger: Logger, log_path: str):
        self._logger = logger
        self._log_path = log_path

        self._pm_capacity = pm_capacity
        self._pm_num = len(self._pm_capacity)

        # LP solver.
        msg = 1 if config.log.stdout_solver_message else 0
        if config.solver == "GLPK":
            self._solver = GLPK(msg=msg)
        elif config.solver == "CBC":
            self._solver = PULP_CBC_CMD(msg=msg)
        else:
            print(f"Solver {config.solver} not added in ILP, choose from [GLPK, CBC]")
            exit(0)

        # For formulation and action application.
        self.plan_window_size = config.plan_window_size
        self.apply_buffer_size = config.apply_buffer_size

        # For performance.
        self.core_upper_ratio = 1 - config.performance.core_safety_remaining_ratio
        self.mem_upper_ratio = 1 - config.performance.mem_safety_remaining_ratio

        # For objective.
        self.successful_allocation_decay = config.objective.successful_allocation_decay
        self.allocation_multiple_core_num = config.objective.allocation_multiple_core_num

        # For logger.
        self.dump_all_solution = config.log.dump_all_solution
        self.dump_infeasible_solution = config.log.dump_infeasible_solution

        # For problem formulation and application
        self.last_solution_env_tick = -1

    def _init_variables(self):
        def _init_with_shape(shape: tuple):
            return np.zeros(shape, dtype=np.int16).tolist()

        # Initialize the PM remaining capacity.
        self._pm_allocated_core = _init_with_shape(shape=(self.plan_window_size, self._pm_num))
        self._pm_allocated_mem = _init_with_shape(shape=(self.plan_window_size, self._pm_num))
        for vm in self._allocated_vm:
            last_tick = min(vm.remaining_lifetime(self._env_tick), self.plan_window_size)
            for t in range(last_tick):
                self._pm_allocated_core[t][vm.pm_idx] += vm.core
                self._pm_allocated_mem[t][vm.pm_idx] += vm.mem

        # Initialize the PM-VM mapping variable.
        self._vm_num = len(self._future_vm_req)
        self._mapping = _init_with_shape(shape=(self._pm_num, self._vm_num))
        for pm_idx in range(self._pm_num):
            for vm_idx in range(self._vm_num):
                self._mapping[pm_idx][vm_idx] = LpVariable(
                    name=f"Place_VM{vm_idx}_{self._future_vm_req[vm_idx].id}_on_PM{pm_idx}",
                    lowBound=0,
                    upBound=1,
                    cat=LpInteger,
                )

    def _add_constraints(self, problem: LpProblem):
        # Mapping limitation: only 1 PM for a VM.
        for vm_idx in range(self._vm_num):
            problem += (
                lpSum(self._mapping[pm_idx][vm_idx] for pm_idx in range(self._pm_num)) <= 1,
                f"Mapping_VM{vm_idx}_to_max_1_PM",
            )

        # PM capacity limitation: core + mem.
        for t in range(self.plan_window_size):
            for pm_idx in range(self._pm_num):
                problem += (
                    lpSum(
                        vm.core * self._mapping[pm_idx][vm_idx]
                        for vm_idx, vm in enumerate(self._future_vm_req)
                        if (vm.arrival_env_tick - self._env_tick <= t and vm.remaining_lifetime(self._env_tick) >= t)
                    )
                    + self._pm_allocated_core[t][pm_idx]
                    <= self._pm_capacity[pm_idx].core * self.core_upper_ratio,
                    f"T{t}_PM{pm_idx}_core_capacity_limit",
                )
                problem += (
                    lpSum(
                        vm.mem * self._mapping[pm_idx][vm_idx]
                        for vm_idx, vm in enumerate(self._future_vm_req)
                        if (vm.arrival_env_tick - self._env_tick <= t and vm.remaining_lifetime(self._env_tick) >= t)
                    )
                    + self._pm_allocated_mem[t][pm_idx]
                    <= self._pm_capacity[pm_idx].mem * self.mem_upper_ratio,
                    f"T{t}_PM{pm_idx}_mem_capacity_limit",
                )

    def _set_objective(self, problem: LpProblem):
        # Reward for successful allocation.
        # allocation_reward = lpSum(
        #     math.pow(self.successful_allocation_decay, self._future_vm_req[vm_idx].arrival_env_tick - self._env_tick)
        #     * lpSum(self._mapping[pm_idx][vm_idx] for pm_idx in range(self._pm_num))
        #     * (self._future_vm_req[vm_idx].core if self.allocation_multiple_core_num else 1)
        #     for vm_idx in range(self._vm_num)
        # )
        allocation_reward = lpSum(
            math.pow(self.successful_allocation_decay, vm_idx)
            * lpSum(self._mapping[pm_idx][vm_idx] for pm_idx in range(self._pm_num))
            * (self._future_vm_req[vm_idx].core if self.allocation_multiple_core_num else 1)
            for vm_idx in range(self._vm_num)
        )

        problem += allocation_reward

    def _formulate_and_solve(self):
        start_time = timeit.default_timer()

        problem = LpProblem(
            name=f"VM_Scheduling_from_tick_{self._env_tick}",
            sense=LpMaximize,
        )
        self._init_variables()
        self._add_constraints(problem=problem)
        self._set_objective(problem=problem)
        problem.solve(self._solver)

        end_time = timeit.default_timer()
        self._logger.info(f"[Timer] {end_time - start_time:.2f} seconds for tick {self._env_tick}.")
        self._logger.info(f"Status: {LpStatus[problem.status]}")
        if self.dump_all_solution or (self.dump_infeasible_solution and problem.status != 1):
            problem.writeLP(os.path.join(self._log_path, f"solution_{self._env_tick}_{LpStatus[problem.status]}.lp"))

        for vm_idx, vm_req in enumerate(self._future_vm_req):
            chosen_pm_idx = -1
            for pm_idx in range(self._pm_num):
                if self._mapping[pm_idx][vm_idx].varValue:
                    chosen_pm_idx = pm_idx
                    break
            self._logger.info(
                f"Solution tick: {self._env_tick}, vm: {self._future_vm_req[vm_idx].id} -> pm: {chosen_pm_idx}",
            )

    def choose_pm(
        self,
        env_tick: int,
        vm_id: int,
        allocated_vm: List[IlpVmInfo],
        future_vm_req: List[IlpVmInfo],
        vm_id_to_idx: dict,
    ) -> int:
        self._env_tick = env_tick

        if self.last_solution_env_tick < 0 or self._env_tick >= self.last_solution_env_tick + self.apply_buffer_size:
            self._allocated_vm = allocated_vm
            self._future_vm_req = future_vm_req
            self._vm_id_to_idx = vm_id_to_idx

            self._formulate_and_solve()
            self.last_solution_env_tick = self._env_tick

        assert vm_id in self._vm_id_to_idx, f"Tick {self._env_tick}, get vm_id {vm_id} not in vm_id_to_idx!"
        vm_idx = self._vm_id_to_idx[vm_id]

        for pm_idx in range(self._pm_num):
            if self._mapping[pm_idx][vm_idx].varValue:
                return pm_idx

        return NOT_ALLOCATE_NOW
