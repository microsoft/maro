# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl import Shaper
from maro.simulator.scenarios.cim.common import Action, ActionType


class CIMActionShaper(Shaper):
    def __init__(self, action_space):
        super().__init__()
        self._action_space = action_space
        self._zero_action_index = action_space.index(0)

    def __call__(self, model_action, decision_event, snapshot_list):
        scope = decision_event.action_scope
        tick = decision_event.tick
        port_idx = decision_event.port_idx
        vessel_idx = decision_event.vessel_idx

        port_empty = snapshot_list["ports"][tick: port_idx: ["empty", "full", "on_shipper", "on_consignee"]][0]
        vessel_remaining_space = snapshot_list["vessels"][tick: vessel_idx: ["empty", "full", "remaining_space"]][2]
        early_discharge = snapshot_list["vessels"][tick:vessel_idx: "early_discharge"][0]
        assert 0 <= model_action < len(self._action_space)
        operation_num = self._action_space[model_action]

        if model_action < self._zero_action_index:
            actual_action = max(round(operation_num * port_empty), -vessel_remaining_space)
            action_type = ActionType.LOAD
        elif model_action > self._zero_action_index:
            plan_action = operation_num * (scope.discharge + early_discharge) - early_discharge
            actual_action = round(plan_action) if plan_action > 0 else round(operation_num * scope.discharge)
            action_type = ActionType.DISCHARGE
        else:
            actual_action = 0
            action_type = None

        return Action(vessel_idx, port_idx, abs(actual_action), action_type)
