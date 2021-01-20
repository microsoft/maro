import pdb
import math
import random

from maro.simulator.scenarios.vm_scheduling.common import Action
from maro.simulator.scenarios.vm_scheduling import AllocateAction, DecisionPayload, PostponeAction

from algorithm import VMSchedulingAgent


class FirstFit(VMSchedulingAgent):
    def __init__(self):
        super().__init__()
        self._pm_list: list[int] = list()

    def choose_action(self, decision_event, env) -> Action:
        env = env
        decision_event = decision_event

        valid_pm_num: int = len(decision_event.valid_pms)

        # Check whether there exists a valid PM.
        if valid_pm_num <= 0:
            # No valid PM now, postpone.
            action: PostponeAction = PostponeAction(
                vm_id=decision_event.vm_id,
                postpone_step=1
            )
        else:
            # Use a valid PM based on its order.
            chosen_idx: int = -1
            for pm_idx in self._pm_list:
                if pm_idx in decision_event.valid_pms:
                    chosen_idx = pm_idx
                    break
            if chosen_idx == -1:
                new_pm_idx: int = len(self._pm_list)
                self._pm_list.append(new_pm_idx)
            # Take action to allocate on the chose PM.
            action: AllocateAction = AllocateAction(
                vm_id=decision_event.vm_id,
                pm_id=chosen_idx
            )

        return action
