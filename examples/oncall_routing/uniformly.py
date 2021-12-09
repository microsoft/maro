# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
from typing import List, Optional

from maro.simulator import Env
from maro.simulator.scenarios.oncall_routing import Order, Route
from maro.simulator.scenarios.oncall_routing.common import Action, OncallRoutingPayload
from maro.utils import set_seeds

set_seeds(0)


def get_random_action(order: Order, routes: List[Route], carriers_in_stop: List[int]) -> Optional[Action]:
    available_route_idxes = [
        idx for idx, route in enumerate(routes)
        if len(route.remaining_plan) >= 2 - carriers_in_stop[route.carrier_idx]
    ]

    if len(available_route_idxes) == 0:
        return None

    route_idx = random.choice(available_route_idxes)
    return Action(
        order_id=order.id,
        route_name=routes[route_idx].name,
        insert_index=random.randint(
            1 - carriers_in_stop[routes[route_idx].carrier_idx],
            len(routes[route_idx].remaining_plan) - 1
        )
    )

# Greedy: assign each on-call order randomly to the available route.
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
        carriers_in_stop: List[int] = env.snapshot_list["carriers"][env.tick::"in_stop"].tolist()

        # Call get_action one by one to get the action without considering segment index
        actions: List[Action] = [get_random_action(order, routes, carriers_in_stop) for order in orders]
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
