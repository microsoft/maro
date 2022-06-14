# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from rule_based_algorithm import RuleBasedAlgorithm

from maro.simulator import Env
from maro.simulator.scenarios.vm_scheduling import AllocateAction, DecisionPayload


class FirstFit(RuleBasedAlgorithm):
    def __init__(self, **kwargs):
        super().__init__()

    def allocate_vm(self, decision_event: DecisionPayload, env: Env) -> AllocateAction:
        # Use a valid PM based on its order.
        chosen_idx: int = decision_event.valid_pms[0]
        # Take action to allocate on the chose PM.
        action: AllocateAction = AllocateAction(
            vm_id=decision_event.vm_id,
            pm_id=chosen_idx,
        )

        return action
