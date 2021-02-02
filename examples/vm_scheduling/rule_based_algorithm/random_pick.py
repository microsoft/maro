import random

from maro.simulator import Env
from maro.simulator.scenarios.vm_scheduling.common import Action
from maro.simulator.scenarios.vm_scheduling import AllocateAction, DecisionPayload, PostponeAction

from algorithm import VMSchedulingAgent


class RandomPick(VMSchedulingAgent):
    def choose_action(self, decision_event: DecisionPayload, env: Env) -> Action:
        valid_pm_num: int = len(decision_event.valid_pms)

        # Check whether there exists a valid PM.
        if valid_pm_num <= 0:
            # No valid PM now, postpone.
            action: PostponeAction = PostponeAction(
                vm_id=decision_event.vm_id,
                postpone_step=1
            )
        else:
            # Random choose a valid PM.
            chosen_idx: int = random.randint(0, valid_pm_num - 1)
            # Take action to allocate on the chosen PM.
            action: AllocateAction = AllocateAction(
                vm_id=decision_event.vm_id,
                pm_id=decision_event.valid_pms[chosen_idx]
            )

        return action
