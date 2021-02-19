from maro.simulator import Env
from maro.simulator.scenarios.vm_scheduling import AllocateAction, DecisionPayload

from rule_based_algorithm import RuleBasedAlgorithm


class RoundRobin(RuleBasedAlgorithm):
    def __init__(self):
        super().__init__()
        self._pm_num: int = None
        self._prev_idx: int = 0

    def allocate_vm(self, decision_event: DecisionPayload, env: Env) -> AllocateAction:
        # Get the number of the PM.
        if self._pm_num is None:
            self._pm_num = self._cal_pm_amount(env)
        # Choose the valid PM which index is next to the previous chose PM's index
        chosen_idx: int = (self._prev_idx + 1) % self._pm_num
        while chosen_idx not in decision_event.valid_pms:
            chosen_idx += 1
            chosen_idx %= self._pm_num
        # Update the prev index
        self._prev_idx = chosen_idx
        # Take action to allocate on the chosen PM.
        action: AllocateAction = AllocateAction(
            vm_id=decision_event.vm_id,
            pm_id=chosen_idx
        )

        return action

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