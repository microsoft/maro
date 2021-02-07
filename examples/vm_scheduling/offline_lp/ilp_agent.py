# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import timeit
from collections import defaultdict
from typing import List, Set

from maro.data_lib import BinaryReader
from maro.simulator.scenarios.vm_scheduling import PostponeAction, AllocateAction
from maro.simulator.scenarios.vm_scheduling.common import Action
from maro.utils import DottableDict, Logger

from common import IlpVmInfo, IlpAllocatedVmInfo, IlpFutureVmInfo, IlpPmCapacity
from vm_scheduling_ilp import NOT_ALLOCATE_NOW, VmSchedulingILP

class IlpAgent():
    def __init__(
        self,
        ilp_config: DottableDict,
        pm_capacity: np.ndarray,
        vm_table_path: str,
        env_start_tick: int,
        env_duration: int,
        simulation_logger: Logger,
        ilp_logger: Logger
    ):
        self._simulation_logger = simulation_logger
        self._ilp_logger = ilp_logger

        pm_capacity: List[IlpPmCapacity] = [IlpPmCapacity(core=pm[0], mem=pm[1]) for pm in pm_capacity]
        self.ilp = VmSchedulingILP(config=ilp_config, pm_capacity=pm_capacity, logger=ilp_logger)
        self.ilp_plan_window_size = ilp_config.plan_window_size

        # Use the vm_item_picker to get the precise vm request info.
        self.vm_reader = BinaryReader(vm_table_path)
        self.vm_item_picker = self.vm_reader.items_tick_picker(
            env_start_tick,
            env_start_tick + env_duration,
            time_unit="s"
        )

        # Used to keep the info already read from the vm_item_picker.
        self.vm_req_dict = defaultdict(list)
        self.allocated_vm_dict = {}
        self.refreshed_allocated_vm_dict = {}

        self.last_env_tick = -1
        self.future_vm_req: List[IlpFutureVmInfo] = []
        self.allocated_vm: List[IlpAllocatedVmInfo] = []

    def choose_action(self, env_tick: int, cur_vm_id: int, live_vm_set_list: List[Set[int]]) -> Action:
        if env_tick != self.last_env_tick:
            self.last_env_tick = env_tick
            self.future_vm_req.clear()
            self.allocated_vm.clear()

            # Read VM data from file.
            for tick in range(env_tick, env_tick + self.ilp_plan_window_size + 1):
                if tick not in self.vm_req_dict:
                    self.vm_req_dict[tick] = [item for item in self.vm_item_picker.items(tick)]

            # Build the future_vm_req list for ILP.
            for tick in range(env_tick, env_tick + self.ilp_plan_window_size + 1):
                for i, vm in enumerate(self.vm_req_dict[tick]):
                    if vm.vm_id == cur_vm_id:
                        self.refreshed_allocated_vm_dict[cur_vm_id] = vm
                    self.future_vm_req.append(
                        IlpFutureVmInfo(
                            core=vm.vm_cpu_cores,
                            mem=vm.vm_memory,
                            remaining_lifetime=vm.vm_lifetime,
                            id=vm.vm_id,
                            arrival_time=env_tick
                        )
                    )

            # Build the allocated_vm list for ILP.
            for pm_idx in range(len(live_vm_set_list)):
                for vm_id in live_vm_set_list[pm_idx]:
                    assert vm_id in self.allocated_vm_dict
                    vm = self.allocated_vm_dict[vm_id]
                    self.refreshed_allocated_vm_dict[vm_id] = vm
                    self.allocated_vm.append(
                        IlpAllocatedVmInfo(
                            pm_idx=pm_idx,
                            core=vm.vm_cpu_cores,
                            mem=vm.vm_memory,
                            remaining_lifetime=vm.vm_lifetime - tick
                        )
                    )

            self.allocated_vm_dict.clear()
            self.allocated_vm_dict = self.refreshed_allocated_vm_dict
            self.refreshed_allocated_vm_dict.clear()

        chosen_pm_idx = self.ilp.choose_pm(env_tick, cur_vm_id, self.allocated_vm, self.future_vm_req)
        self._simulation_logger.info(f"vm: {cur_vm_id} -> pm: {chosen_pm_idx}")

        if chosen_pm_idx == NOT_ALLOCATE_NOW:
            return PostponeAction(vm_id=cur_vm_id, postpone_step=1)
        else:
            return AllocateAction(vm_id=cur_vm_id, pm_id=chosen_pm_idx)
