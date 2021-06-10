# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.simulator.scenarios.vm_scheduling.common import Action
from maro.simulator.scenarios.vm_scheduling import AllocateAction, DecisionPayload, PostponeAction


class VMAction(object):
    def __init__(self, pm_num):
        self._pm_num = pm_num

    def __call__(self, model_action, decision_event):
        if model_action == self._pm_num:
            action: PostponeAction = PostponeAction(
                vm_id=decision_event.vm_id,
                postpone_step=1
            )
        else:
            action: AllocateAction = AllocateAction(
                vm_id=decision_event.vm_id,
                pm_id=model_action
            )
        return action
