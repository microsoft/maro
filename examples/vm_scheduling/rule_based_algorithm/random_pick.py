import pdb
import random

from maro.simulator.scenarios.vm_scheduling.common import Action
from maro.simulator.scenarios.vm_scheduling import AllocateAction, DecisionPayload, PostponeAction

from algorithm import Algorithm


class RandomPick(Algorithm):
    def __init__(self):
        super().__init__()

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
            # Random choose a valid PM.
            chosen_idx: int = random.randint(0, valid_pm_num - 1)
            # Take action to allocate on the chosen PM.
            action: AllocateAction = AllocateAction(
                vm_id=self.decision_event.vm_id,
                pm_id=self.decision_event.valid_pms[chosen_idx]
            )

        return action
