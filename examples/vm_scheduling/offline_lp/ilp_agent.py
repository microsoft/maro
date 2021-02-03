# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import timeit
from collections import defaultdict
from typing import List, Set

from maro.data_lib import BinaryReader
from maro.simulator.scenarios.vm_scheduling import PostponeAction, AllocateAction
from maro.simulator.scenarios.vm_scheduling.common import Action
from maro.utils import DottableDict

from common import IlpVmInfo, IlpAllocatedVmInfo, IlpFutureVmInfo, IlpPmCapacity
from vm_scheduling_ilp import NOT_ALLOCATE_NOW, VmSchedulingILP

class IlpAgent():
    def __init__(
        self,
        ilp_config: DottableDict,
        pm_capacity: np.ndarray,
        vm_table_path: str,
        env_start_tick: int,
        env_duration: int
    ):
        pm_capacity: List[IlpPmCapacity] = [IlpPmCapacity(core=pm[0], mem=pm[1]) for pm in pm_capacity]
        self.ilp = VmSchedulingILP(config=ilp_config, pm_capacity=pm_capacity)
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
        self.vm_id_to_tick_idx = {}
        self.current_tick = -1
        self.current_tick_idx = -1

    def choose_action(self, env_tick: int, cur_vm_id: int, live_vm_set_list: List[Set[int]]) -> Action:
        # TODO: Add data erasure of useless old data.
        for tick in range(env_tick, env_tick + self.ilp_plan_window_size + 1):
            if tick not in self.vm_req_dict:
                self.vm_req_dict[tick] = [item for item in self.vm_item_picker.items(tick)]

        if env_tick > self.current_tick:
            self.current_tick = env_tick
            self.current_tick_idx = -1

        vm_req: IlpVmInfo = None
        while self.current_tick_idx < len(self.vm_req_dict[env_tick]):
            self.current_tick_idx += 1
            vm = self.vm_req_dict[env_tick][self.current_tick_idx]
            if vm.vm_id == cur_vm_id:
                vm_req = IlpVmInfo(core=vm.vm_cpu_cores, mem=vm.vm_memory, remaining_lifetime=vm.vm_lifetime)
                self.vm_id_to_tick_idx[vm.vm_id] = (env_tick, self.current_tick_idx)
                break
        assert vm_req is not None

        future_vm_req: List[IlpFutureVmInfo] = []
        idx = self.current_tick_idx + 1
        while idx < len(self.vm_req_dict[env_tick]):
            vm = self.vm_req_dict[env_tick][idx]
            future_vm_req.append(
                IlpFutureVmInfo(
                    core=vm.vm_cpu_cores,
                    mem=vm.vm_memory,
                    remaining_lifetime=vm.vm_lifetime,
                    arrival_time=env_tick
                )
            )
            idx += 1

        for tick in range(env_tick + 1, env_tick + self.ilp_plan_window_size + 1):
            for vm in self.vm_req_dict[tick]:
                future_vm_req.append(
                    IlpFutureVmInfo(
                        core=vm.vm_cpu_cores,
                        mem=vm.vm_memory,
                        remaining_lifetime=vm.vm_lifetime,
                        arrival_time=env_tick
                    )
                )

        allocated_vm: List[IlpAllocatedVmInfo] = []
        for pm_idx in range(len(live_vm_set_list)):
            for vm_id in live_vm_set_list[pm_idx]:
                assert vm_id in self.vm_id_to_tick_idx
                tick, idx = self.vm_id_to_tick_idx[vm_id]
                vm = self.vm_req_dict[tick][idx]
                allocated_vm.append(
                    IlpAllocatedVmInfo(
                        pm_idx=pm_idx,
                        core=vm.vm_cpu_cores,
                        mem=vm.vm_memory,
                        remaining_lifetime=vm.vm_lifetime - tick
                    )
                )

        start_time = timeit.default_timer()
        chosen_pm_idx = self.ilp.choose_pm(env_tick, vm_req, allocated_vm, future_vm_req)
        end_time = timeit.default_timer()
        print(f"vm: {cur_vm_id} -> pm: {chosen_pm_idx}")
        if chosen_pm_idx == NOT_ALLOCATE_NOW:
            return PostponeAction(vm_id=cur_vm_id, postpone_step=1)
        else:
            return AllocateAction(vm_id=cur_vm_id, pm_id=chosen_pm_idx)
