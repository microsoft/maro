# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Optional

from maro.simulator import Env
from maro.simulator.scenarios.oncall_routing import Order, Route
from maro.simulator.scenarios.oncall_routing.common import Action, OncallRoutingPayload
from maro.simulator.scenarios.oncall_routing.utils import geo_distance_meter
from maro.utils import set_seeds

set_seeds(0)


def get_greedy_action(order: Order, routes: List[Route], carriers_in_stop: List[bool]) -> Optional[Action]:
    min_distance: float = float("inf")
    route_name: str = None
    insert_index: int = -1
    for route in routes:
        start = 0 if carriers_in_stop[route.carrier_idx] else 1
        for idx in range(start, len(route.remaining_plan)):
            distance = geo_distance_meter(order.coord, route.remaining_plan[idx].order.coord)
            if distance < min_distance:
                min_distance = distance
                route_name = route.name
                insert_index = idx

    if route_name is None:
        return None

    return Action(order_id=order.id, route_name=route_name, insert_index=insert_index)

# Greedy: assign each on-call order to the closest stop on existing route.
if __name__ == "__main__":
    env = Env(
        scenario="oncall_routing", topology="example", start_tick=0, durations=1440,
    )

    # TODO: check the reset functionality
    # env.reset()
    metrics, decision_event, is_done = env.step(None)
    while not is_done:
        assert isinstance(decision_event, OncallRoutingPayload)
        orders = decision_event.oncall_orders
        routes = decision_event.routes_info
        carriers_in_stop: List[bool] = (env.snapshot_list["carriers"][env.tick::"in_stop"] == 1).tolist()

        # Call get_action one by one to get the action without considering segment index
        actions: List[Action] = [get_greedy_action(order, routes, carriers_in_stop) for order in orders]
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
