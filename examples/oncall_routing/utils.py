# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

from maro.simulator.scenarios.oncall_routing.common import Action


def refresh_segment_index(actions: List[Action]) -> List[Action]:
    # Add segment index if multiple orders are sharing a same insert index.

    def _is_equal_segment(action1: Action, action2: Action) -> bool:
        return (action1.route_name, action1.insert_index) == (action2.route_name, action2.insert_index)

    actions.sort(key=lambda action: (action.route_name, action.insert_index))
    segment_index = 0
    for idx in range(len(actions) - 1):
        if _is_equal_segment(actions[idx], actions[idx + 1]):
            segment_index += 1
            actions[idx + 1].in_segment_order = segment_index
        else:
            segment_index = 0

    return actions
