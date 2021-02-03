from maro.simulator import Env
from maro.simulator.scenarios.vm_scheduling.common import Action
from maro.simulator.scenarios.vm_scheduling import AllocateAction, DecisionPayload, PostponeAction

from algorithm import VMSchedulingAgent


class FirstFit(VMSchedulingAgent):
    def __init__(self):
        super().__init__()
        self._pm_list: list[int] = list()

    def allocate_vm(self, decision_event: DecisionPayload, env: Env) -> AllocateAction:
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
