import pdb
import random
import numpy as np

from maro.simulator.scenarios.vm_scheduling import AllocateAction, DecisionPayload, PostponeAction


class BinPacking(object):
    def __init__(self, pm_num, pm_cpu_core_num):
        self.pm_num = pm_num
        self.pm_cpu_core_num = pm_cpu_core_num

        self.decision_event = None
        self.env = None

        self.bin = dict()
        self.bin_length = [0] * self.pm_cpu_core_num

    def _init_bin(self):
        self.bin_length = [0] * (self.pm_cpu_core_num+1)
        for i in range(self.pm_cpu_core_num+1):
            self.bin[i] = list()

    def get_action(self, decision_event, env):
        self.decision_event = decision_event
        self.env = env

        valid_pm_num: int = len(self.decision_event.valid_pms)
        print(valid_pm_num)

        if valid_pm_num <= 0:
            # No valid PM now, postpone.
            action: PostponeAction = PostponeAction(
                vm_id=self.decision_event.vm_id,
                postpone_step=1
            )
        else:
            self._init_bin()

            total_pm_info = self.env.snapshot_list["pms"][
                                self.env.frame_index:list(range(self.pm_num)):["cpu_cores_capacity", "cpu_cores_allocated"]
                            ].reshape(-1, 2)
            cpu_cores_remaining = total_pm_info[:, 0] - total_pm_info[:, 1]

            for i, cpu_core in enumerate(cpu_cores_remaining):
                self.bin[int(cpu_core)].append(i)
                self.bin_length[int(cpu_core)] += 1

            minimal_var = np.inf
            vm_info = [self.decision_event.vm_cpu_cores_requirement, self.decision_event.vm_memory_requirement]

            chosen_idx = 0
            for i in range(len(self.bin)):
                if self.bin_length[i] != 0 and i >= vm_info[0]:
                    self.bin_length[i] -= 1
                    self.bin_length[i - vm_info[0]] += 1
                    var = np.var(np.array(self.bin_length))
                    if minimal_var > var:
                        minimal_var = var
                        chosen_idx = random.sample(self.bin[i], 1)[0]
                    self.bin_length[i] += 1
                    self.bin_length[i - vm_info[0]] -= 1

            # Take action to allocate on the closet pm.
            action: AllocateAction = AllocateAction(
                vm_id=self.decision_event.vm_id,
                pm_id=chosen_idx
            )

        return action
