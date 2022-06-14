# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import Counter, defaultdict
from typing import List, Set

import numpy as np
from common import IlpPmCapacity, IlpVmInfo
from vm_scheduling_ilp import NOT_ALLOCATE_NOW, VmSchedulingILP

from maro.data_lib import BinaryReader
from maro.simulator.scenarios.vm_scheduling import AllocateAction, PostponeAction
from maro.simulator.scenarios.vm_scheduling.common import Action
from maro.utils import DottableDict, Logger


class IlpAgent:
    def __init__(
        self,
        ilp_config: DottableDict,
        pm_capacity: np.ndarray,
        vm_table_path: str,
        env_start_tick: int,
        env_duration: int,
        simulation_logger: Logger,
        ilp_logger: Logger,
        log_path: str,
    ):
        self._simulation_logger = simulation_logger
        self._ilp_logger = ilp_logger

        self._allocation_counter = Counter()

        pm_capacity: List[IlpPmCapacity] = [IlpPmCapacity(core=pm[0], mem=pm[1]) for pm in pm_capacity]
        self.ilp = VmSchedulingILP(config=ilp_config, pm_capacity=pm_capacity, logger=ilp_logger, log_path=log_path)
        self.ilp_plan_window_size = ilp_config.plan_window_size
        self.ilp_apply_buffer_size = ilp_config.apply_buffer_size

        # Use the vm_item_picker to get the precise vm request info.
        self.vm_reader = BinaryReader(vm_table_path)
        self.vm_item_picker = self.vm_reader.items_tick_picker(
            env_start_tick,
            env_start_tick + env_duration,
            time_unit="s",
        )

        # Used to keep the info already read from the vm_item_picker.
        self.vm_req_dict = defaultdict(list)
        self.env_tick_in_vm_req_dict = []
        self.allocated_vm_dict = {}
        self.refreshed_allocated_vm_dict = {}

        self.last_solution_env_tick = -1
        self._vm_id_to_idx = {}
        self.future_vm_req: List[IlpVmInfo] = []
        self.allocated_vm: List[IlpVmInfo] = []

    def choose_action(self, env_tick: int, cur_vm_id: int, live_vm_set_list: List[Set[int]]) -> Action:
        # Formulate and solve only when the new request goes beyond the apply buffer size of last ILP solution.
        if self.last_solution_env_tick < 0 or env_tick >= self.last_solution_env_tick + self.ilp_apply_buffer_size:
            self.last_solution_env_tick = env_tick
            self._vm_id_to_idx = {}
            self.future_vm_req.clear()
            self.allocated_vm.clear()

            # To clear the outdated vm_req_dict data.
            pop_num = 0
            for i, tick in enumerate(self.env_tick_in_vm_req_dict):
                if tick < env_tick:
                    self.vm_req_dict.pop(tick)
                    pop_num += 1
                else:
                    break
            self.env_tick_in_vm_req_dict = self.env_tick_in_vm_req_dict[pop_num:]

            # Read VM data from file.
            for tick in range(env_tick, env_tick + self.ilp_plan_window_size + 1):
                if tick not in self.vm_req_dict:
                    self.env_tick_in_vm_req_dict.append(tick)
                    self.vm_req_dict[tick] = [item for item in self.vm_item_picker.items(tick)]

            # Build the future_vm_req list for ILP.
            for tick in range(env_tick, env_tick + self.ilp_plan_window_size + 1):
                for vm in self.vm_req_dict[tick]:
                    vmInfo = IlpVmInfo(
                        id=vm.vm_id,
                        core=vm.vm_cpu_cores,
                        mem=vm.vm_memory,
                        lifetime=vm.vm_lifetime,
                        arrival_env_tick=tick,
                    )
                    if tick < env_tick + self.ilp_apply_buffer_size:
                        self.refreshed_allocated_vm_dict[vm.vm_id] = vmInfo
                    self._vm_id_to_idx[vm.vm_id] = len(self.future_vm_req)
                    self.future_vm_req.append(vmInfo)

            # Build the allocated_vm list for ILP.
            for pm_idx in range(len(live_vm_set_list)):
                for vm_id in live_vm_set_list[pm_idx]:
                    assert vm_id in self.allocated_vm_dict, f"ILP agent: vm_id {vm_id} not in allocated_vm_dict"
                    vm = self.allocated_vm_dict[vm_id]
                    vm.pm_idx = pm_idx
                    self.refreshed_allocated_vm_dict[vm_id] = vm
                    self.allocated_vm.append(vm)

            self.allocated_vm_dict.clear()
            self.allocated_vm_dict = self.refreshed_allocated_vm_dict
            self.refreshed_allocated_vm_dict = {}

        # Choose action by ILP, may trigger a new formulation and solution,
        # may directly return the decision if the cur_vm_id is still in the apply buffer size of last solution.
        chosen_pm_idx = self.ilp.choose_pm(
            env_tick,
            cur_vm_id,
            self.allocated_vm,
            self.future_vm_req,
            self._vm_id_to_idx,
        )
        self._simulation_logger.info(f"tick: {env_tick}, vm: {cur_vm_id} -> pm: {chosen_pm_idx}")

        if chosen_pm_idx == NOT_ALLOCATE_NOW:
            return PostponeAction(vm_id=cur_vm_id, postpone_step=1)
        else:
            self._allocation_counter[self.future_vm_req[self._vm_id_to_idx[cur_vm_id]].core] += 1
            return AllocateAction(vm_id=cur_vm_id, pm_id=chosen_pm_idx)

    def report_allocation_summary(self):
        self._simulation_logger.info(f"Allocation Counter(#core, #req): {sorted(self._allocation_counter.items())}")
