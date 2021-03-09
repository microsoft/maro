# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl import Shaper
from maro.simulator.scenarios.vm_scheduling import AllocateAction


class VMActionShaper(Shaper):
    def __init__(self):
        super().__init__()

    def __call__(self, model_action, decision_event):
        action: AllocateAction = AllocateAction(
            vm_id=decision_event.vm_id,
            pm_id=model_action
        )
        return action
