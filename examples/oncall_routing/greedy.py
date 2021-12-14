# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List, Optional

from maro.simulator import Env
from maro.simulator.scenarios.oncall_routing import Order, Route
from maro.simulator.scenarios.oncall_routing.common import Action, OncallRoutingPayload
from maro.simulator.scenarios.oncall_routing.utils import geo_distance_meter
from maro.utils import set_seeds

set_seeds(0)


def get_greedy_action(
    order: Order,
    route_meta_info_dict: dict,
    route_plan_dict: Dict[str, List[Order]],
    carriers_in_stop: List[bool]
) -> Optional[Action]:
    min_distance: float = float("inf")
    insert_index: int = -1
    chosen_route_name: Optional[str] = None

    for route_name in route_meta_info_dict:
        carrier_idx = route_meta_info_dict[route_name]["carrier_idx"]
        start = 0 if carriers_in_stop[carrier_idx] else 1
        plan = route_plan_dict[route_name]

        for i in range(start, len(plan)):
            distance = geo_distance_meter(order.coord, plan[i].coord)
            if distance < min_distance:
                min_distance = distance
                chosen_route_name = route_name
                insert_index = i

    if chosen_route_name is None:
        return None

    return Action(order_id=order.id, route_name=chosen_route_name, insert_index=insert_index)


# Greedy: assign each on-call order to the closest stop on existing route.
if __name__ == "__main__":
    env = Env(
        scenario="oncall_routing", topology="example", start_tick=0, durations=1440,
    )

    # TODO: check the reset functionality
    env.reset(keep_seed=True)
    metrics, decision_event, is_done = env.step(None)
    while not is_done:
        assert isinstance(decision_event, OncallRoutingPayload)
        orders = decision_event.oncall_orders
        route_plan_dict = decision_event.route_plan_dict
        carriers_in_stop: List[bool] = (env.snapshot_list["carriers"][env.tick::"in_stop"] == 1).tolist()
        route_meta_info_dict = decision_event.route_meta_info_dict

        # Call get_action one by one to get the action without considering segment index
        actions: List[Action] = [get_greedy_action(
            order, route_meta_info_dict, route_plan_dict, carriers_in_stop
        ) for order in orders]
        actions = [action for action in actions if action]
        # Add segment index if multiple orders are share
        actions = sorted(actions, key=lambda action: (action.route_name, action.insert_index))
        segment_index = 0
        for idx in range(len(actions) - 1):
            if (
                actions[idx + 1].route_name == actions[idx].route_name
                and actions[idx + 1].insert_index == actions[idx].insert_index
            ):
                segment_index += 1
                actions[idx + 1].in_segment_order = segment_index
            else:
                segment_index = 0

        metrics, decision_event, is_done = env.step(actions)

    print(metrics)
