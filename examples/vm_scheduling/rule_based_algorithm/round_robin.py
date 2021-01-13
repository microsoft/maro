import pdb
import random

from maro.simulator.scenarios.vm_scheduling.common import Action
from maro.simulator.scenarios.vm_scheduling import AllocateAction, DecisionPayload, PostponeAction

from algorithm import Algorithm


class RoundRobin(Algorithm):
    def __init__(
        self,
        pm_num: int = 0
    ):
        super().__init__()
        self.pm_num: int = pm_num

    def get_action(self, decision_event, env, prev_idx=None) -> Action:
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
            # Choose the valid PM which index is next to the previous chose PM's index
            chosen_idx: int = 0
            if prev_idx is not None:
                chosen_idx = (prev_idx + 1) % self.pm_num
                while True:
                    if chosen_idx in self.decision_event.valid_pms:
                        break
                    else:
                        chosen_idx += 1
                        chosen_idx %= self.pm_num

            # Take action to allocate on the chosen PM.
            action: AllocateAction = AllocateAction(
                vm_id=self.decision_event.vm_id,
                pm_id=chosen_idx
            )

        return action, chosen_idx
