import random
import numpy as np

from maro.simulator import Env
from maro.simulator.scenarios.vm_scheduling import AllocateAction, DecisionPayload

from rule_based_algorithm import RuleBasedAlgorithm


class BinPacking(RuleBasedAlgorithm):
    def __init__(
        self, 
        pm_num: int, 
        pm_cpu_core_num: int
    ):
        super().__init__()
        self._pm_num: int = pm_num
        self._pm_cpu_core_num: int = pm_cpu_core_num

    def _init_bin(self):
        self._bins = [[] for _ in range(self._pm_cpu_core_num + 1)]
        self._bin_size = [0] * (self._pm_cpu_core_num + 1)

    def allocate_vm(self, decision_event: DecisionPayload, env: Env) -> AllocateAction:
        self._init_bin()

        total_pm_info = env.snapshot_list["pms"][
            env.frame_index::["cpu_cores_capacity", "cpu_cores_allocated"]
        ].reshape(-1, 2)
        cpu_cores_remaining = total_pm_info[:, 0] - total_pm_info[:, 1]

        for i, cpu_core in enumerate(cpu_cores_remaining):
            self._bins[int(cpu_core)].append(i)
            self._bin_size[int(cpu_core)] += 1

        minimal_var = np.inf
        cores_need = decision_event.vm_cpu_cores_requirement

        chosen_idx = 0
        for remaining_cores in range(cores_need, len(self._bins)):
            if self._bin_size[remaining_cores] != 0:
                self._bin_size[remaining_cores] -= 1
                self._bin_size[remaining_cores - cores_need] += 1
                var = np.var(np.array(self._bin_size))
                if minimal_var > var:
                    minimal_var = var
                    chosen_idx = random.choice(self._bins[remaining_cores])
                self._bin_size[remaining_cores] += 1
                self._bin_size[remaining_cores - cores_need] -= 1

        # Take action to allocate on the closet pm.
        action: AllocateAction = AllocateAction(
            vm_id=decision_event.vm_id,
            pm_id=chosen_idx
        )

        return action
