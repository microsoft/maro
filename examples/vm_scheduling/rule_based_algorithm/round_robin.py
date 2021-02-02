from maro.simulator import Env
from maro.simulator.scenarios.vm_scheduling.common import Action
from maro.simulator.scenarios.vm_scheduling import AllocateAction, DecisionPayload, PostponeAction

from algorithm import VMSchedulingAgent


class RoundRobin(VMSchedulingAgent):
    def __init__(
        self,
        pm_num: int = 0
    ):
        super().__init__()
        self._pm_num: int = pm_num
        self._prev_idx: int = 0

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
            # Choose the valid PM which index is next to the previous chose PM's index
            chosen_idx: int = (self._prev_idx + 1) % self._pm_num
            while True:
                if chosen_idx in decision_event.valid_pms:
                    break
                else:
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
