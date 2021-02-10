import random
import numpy as np
from yaml import safe_load

from maro.simulator import Env
from maro.simulator.scenarios.vm_scheduling import AllocateAction, DecisionPayload
from maro.utils.utils import convert_dottable

from rule_based_algorithm import RuleBasedAlgorithm


class BinPacking(RuleBasedAlgorithm):
    def __init__(self):
        super().__init__()
        self._pm_num: int = None
        self._pm_cpu_core_num: int = None
        self._max_cpu_oversubscription_rate: float = None

    def _init_bin(self):
        self._bins = [[] for _ in range(self._pm_cpu_core_num + 1)]
        self._bin_size = [0] * (self._pm_cpu_core_num + 1)

    def allocate_vm(self, decision_event: DecisionPayload, env: Env) -> AllocateAction:
        # Get the number of PM, maximum CPU core and max cpu oversubscription rate.
        if self._max_cpu_oversubscription_rate is None:
            self._pm_num = self._cal_pm_amount(env)
            self._max_cpu_oversubscription_rate = env.configs.MAX_CPU_OVERSUBSCRIPTION_RATE
            self._pm_cpu_core_num = int(self._cal_max_pm_cpu_core(env) * self._max_cpu_oversubscription_rate)

        total_pm_info = env.snapshot_list["pms"][
                        env.frame_index::["cpu_cores_capacity", "cpu_cores_allocated"]
                        ].reshape(-1, 2)

        # Initialize the bin.
        self._init_bin()

        cpu_cores_remaining = total_pm_info[:, 0] * self._max_cpu_oversubscription_rate - total_pm_info[:, 1]

        for i, cpu_core in enumerate(cpu_cores_remaining):
            self._bins[int(cpu_core)].append(i)
            self._bin_size[int(cpu_core)] += 1

        # Choose a PM that minimize the variance of the PM number in each bin.
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

        # Take action to allocate on the chosen pm.
        action: AllocateAction = AllocateAction(
            vm_id=decision_event.vm_id,
            pm_id=chosen_idx
        )

        return action

    def _cal_max_pm_cpu_core(self, env: Env) -> int:
        cpu_core: int = 0

        for pm_list in self._find_item(key="pm", dictionary=env.configs.components):
            for pm in pm_list:
                if "cpu" in pm.keys():
                    cpu_core = max(cpu_core, pm["cpu"])

        return cpu_core

    def _cal_pm_amount(self, env: Env) -> int:
        # Cluster amount dict.
        cluster_amount_dict = {}
        for cluster_list in self._find_item(key="cluster", dictionary=env.configs.architecture):
            for cluster in cluster_list:
                cluster_amount_dict[cluster['type']] = (
                        cluster_amount_dict.get(cluster['type'], 0) + cluster['cluster_amount']
                )

        # Rack amount dict.
        rack_amount_dict = {}
        for cluster_list in self._find_item(key="cluster", dictionary=env.configs.components):
            for cluster in cluster_list:
                for rack in cluster['rack']:
                    rack_amount_dict[rack['rack_type']] = (
                            rack_amount_dict.get(rack['rack_type'], 0)
                            + cluster_amount_dict[cluster['type']] * rack['rack_amount']
                    )
        # PM amount dict.
        pm_amount_dict = {}
        for rack in env.configs.components.rack:
            for pm in rack['pm']:
                pm_amount_dict[pm['pm_type']] = (
                        pm_amount_dict.get(pm['pm_type'], 0)
                        + rack_amount_dict[rack['type']] * pm['pm_amount']
                )
        # Summation of pm amount.
        amount: int = sum(value for value in pm_amount_dict.values())

        return amount

    def _find_item(self, key: str, dictionary: dict) -> int:
        for k, v in dictionary.items():
            if k == key:
                yield v
            elif isinstance(v, list):
                for item in v:
                    for result in self._find_item(key, item):
                        yield result
            elif isinstance(v, dict):
                for result in self._find_item(key, v):
                    yield result