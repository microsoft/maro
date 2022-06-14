# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random

from rule_based_algorithm import RuleBasedAlgorithm

from maro.simulator import Env
from maro.simulator.scenarios.vm_scheduling import AllocateAction, DecisionPayload


class RandomPick(RuleBasedAlgorithm):
    def __init__(self, **kwargs):
        super().__init__()

    def allocate_vm(self, decision_event: DecisionPayload, env: Env) -> AllocateAction:
        valid_pm_num: int = len(decision_event.valid_pms)
        # Random choose a valid PM.
        chosen_idx: int = random.randint(0, valid_pm_num - 1)
        # Take action to allocate on the chosen PM.
        action: AllocateAction = AllocateAction(
            vm_id=decision_event.vm_id,
            pm_id=decision_event.valid_pms[chosen_idx],
        )

        return action
