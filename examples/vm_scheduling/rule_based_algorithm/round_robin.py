# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from rule_based_algorithm import RuleBasedAlgorithm

from maro.simulator import Env
from maro.simulator.scenarios.vm_scheduling import AllocateAction, DecisionPayload


class RoundRobin(RuleBasedAlgorithm):
    def __init__(self, **kwargs):
        super().__init__()
        self._prev_idx: int = 0
        self._pm_num: int = (
            kwargs["env"].snapshot_list["pms"][kwargs["env"].frame_index :: ["cpu_cores_capacity"]].shape[0]
        )

    def allocate_vm(self, decision_event: DecisionPayload, env: Env) -> AllocateAction:
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
            pm_id=chosen_idx,
        )

        return action
