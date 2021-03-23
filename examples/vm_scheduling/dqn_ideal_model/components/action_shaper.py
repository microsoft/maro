# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl import Shaper
from maro.simulator.scenarios.vm_scheduling.common import Action
from maro.simulator.scenarios.vm_scheduling import AllocateAction, DecisionPayload, PostponeAction


class VMActionShaper(Shaper):
    def __init__(self):
        super().__init__()

    def __call__(self, model_action, decision_event):
        if model_action is None:
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
