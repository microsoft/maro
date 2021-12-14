# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
from typing import Dict, List, Optional

from maro.simulator import Env
from maro.simulator.scenarios.oncall_routing import Order
from maro.simulator.scenarios.oncall_routing.common import Action, OncallRoutingPayload
from maro.utils import set_seeds

set_seeds(0)


def get_random_action(
    order: Order,
    route_meta_info_dict: dict,
    route_plan_dict: Dict[str, List[Order]],
    carriers_in_stop: List[bool]
) -> Optional[Action]:
    available_route_names = []
    for route_name in route_plan_dict:
        plan = route_plan_dict[route_name]
        carrier_idx = route_meta_info_dict[route_name]["carrier_idx"]
        if len(plan) >= 2 - (1 if carriers_in_stop[carrier_idx] else 0):
            available_route_names.append(route_name)

    if len(available_route_names) == 0:
        return None

    chosen_route_name = random.choice(available_route_names)
    chosen_carrier_idx = route_meta_info_dict[chosen_route_name]["carrier_idx"]
    return Action(
        order_id=order.id,
        route_name=chosen_route_name,
        insert_index=random.randint(
            1 - carriers_in_stop[chosen_carrier_idx],
            len(route_plan_dict[chosen_route_name]) - 1
        )
    )


# Random: assign each on-call order randomly to the available route.
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
        print(f"Processing {len(orders)} orders at tick {env.tick}.")
        actions: List[Action] = [get_random_action(
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
