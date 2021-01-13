import pdb
import math
import random

from maro.simulator.scenarios.vm_scheduling.common import Action
from maro.simulator.scenarios.vm_scheduling import AllocateAction, DecisionPayload, PostponeAction

from algorithm import Algorithm


class FirstFit(Algorithm):
    def __init__(self):
        super().__init__()
        self.pm_list: list[int] = list()

    def get_action(self, decision_event, env) -> Action:
        self.env = env
        self.decision_event = decision_event

        valid_pm_num: int = len(self.decision_event.valid_pms)

        # Check whether there exists a valid PM.
        if valid_pm_num <= 0:
            # No valid PM now, postpone.
            action: PostponeAction = PostponeAction(
                vm_id=self.decision_event.vm_id,
                postpone_step=1
            )
        else:
            # Use a valid PM based on its order.
            chosen_idx: int = -1
            for pm_idx in self.pm_list:
                if pm_idx in self.decision_event.valid_pms:
                    chosen_idx = pm_idx
                    break
            if chosen_idx == -1:
                new_pm_idx: int = len(self.pm_list)
                self.pm_list.append(new_pm_idx)
            # Take action to allocate on the chose PM.
            action: AllocateAction = AllocateAction(
                vm_id=self.decision_event.vm_id,
                pm_id=chosen_idx
            )

        return action
