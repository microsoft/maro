# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

from maro.simulator.scenarios.oncall_routing.common import Action, AllocateAction, PostponeAction


def refresh_segment_index(actions: List[Action]) -> List[Action]:
    # Add segment index if multiple orders are sharing a same insert index.

    def _is_equal_segment(action1: Action, action2: Action) -> bool:
        return (action1.route_name, action1.insert_index) == (action2.route_name, action2.insert_index)

    postpone_actions = [action for action in actions if isinstance(action, PostponeAction)]
    allocate_actions = [action for action in actions if isinstance(action, AllocateAction)]

    allocate_actions.sort(key=lambda action: (action.route_name, action.insert_index))
    segment_index = 0
    for idx in range(len(allocate_actions) - 1):
        if _is_equal_segment(allocate_actions[idx], allocate_actions[idx + 1]):
            segment_index += 1
            allocate_actions[idx + 1].in_segment_order = segment_index
        else:
            segment_index = 0

    return allocate_actions + postpone_actions
