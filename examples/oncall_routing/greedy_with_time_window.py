# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List, Optional

from maro.simulator import Env
from maro.simulator.scenarios.oncall_routing import Coordinate
from maro.simulator.scenarios.oncall_routing.common import Action, AllocateAction, OncallRoutingPayload, PostponeAction
from maro.utils import set_seeds

from examples.oncall_routing.utils import refresh_segment_index

set_seeds(0)


def _get_actions(running_env: Env, event: OncallRoutingPayload) -> List[Action]:
    tick = running_env.tick
    frame_index = running_env.business_engine.frame_index(running_env.tick)
    oncall_orders = event.oncall_orders
    route_meta_info_dict = event.route_meta_info_dict
    route_plan_dict = event.route_plan_dict
    carriers_in_stop: List[bool] = (running_env.snapshot_list["carriers"][frame_index::"in_stop"] == 1).tolist()
    est_duration_predictor = event.estimated_duration_predictor

    route_original_indexes = {}
    for route_name in route_meta_info_dict:
        num_order = len(route_plan_dict[route_name])
        route_original_indexes[route_name] = list(range(num_order))

    actions = []
    for oncall_order in oncall_orders:
        # Best result with violating time windows
        min_duration_violate = float("inf")
        chosen_route_name_violate: Optional[str] = None
        insert_idx_violate = -1

        # Best result without violating any time windows
        min_duration_no_violate = float("inf")
        chosen_route_name_no_violate: Optional[str] = None
        insert_idx_no_violate = -1

        for route_name, meta in route_meta_info_dict.items():
            carrier_idx = meta["carrier_idx"]
            estimated_next_departure_tick: int = meta["estimated_next_departure_tick"]
            planned_orders = route_plan_dict[route_name]

            for i, planned_order in enumerate(planned_orders):  # To traverse the insert index
                if i == 0 and not carriers_in_stop[carrier_idx]:
                    continue

                duration = None

                # Check if it will violate any time window
                is_time_valid = True
                cur_tick = tick
                for j in range(len(planned_orders)):  # Simulate all orders
                    if j == i:  # If we need to insert the oncall order before the j'th planned order
                        if j == 0:  # Carrier in stop. Insert before the first stop.
                            current_staying_stop_coordinate: Coordinate = meta["current_staying_stop_coordinate"]
                            cur_tick += estimated_next_departure_tick
                            cur_tick += est_duration_predictor.predict(  # Current stop => oncall order
                                cur_tick, current_staying_stop_coordinate, oncall_order.coord)
                        else:
                            cur_tick += est_duration_predictor.predict(  # Last planned order => oncall order
                                cur_tick, planned_orders[j - 1].coord, oncall_order.coord
                            )

                        duration = est_duration_predictor.predict(cur_tick, oncall_order.coord, planned_order.coord)

                        # Check if violate the oncall order time window or not
                        if not oncall_order.open_time <= cur_tick <= oncall_order.close_time:
                            is_time_valid = False
                            break

                        cur_tick += duration  # Oncall order => current planned order

                    else:
                        if j == 0:
                            # Current position (on the way or in a stop) => first planned order
                            if carriers_in_stop[carrier_idx]:
                                cur_tick = estimated_next_departure_tick
                            cur_tick += meta["estimated_duration_to_the_next_stop"]
                        else:
                            # Last planned order => current planned order
                            cur_tick += est_duration_predictor.predict(
                                cur_tick, planned_orders[j - 1].coord, planned_orders[j].coord
                            )

                    # Violate current planned order time window
                    if all([
                        duration is not None,
                        not planned_orders[j].open_time <= cur_tick <= planned_orders[j].close_time
                    ]):
                        is_time_valid = False
                        break

                if not duration:
                    continue

                if is_time_valid:
                    if duration < min_duration_no_violate:
                        min_duration_no_violate = duration
                        chosen_route_name_no_violate = route_name
                        insert_idx_no_violate = i
                else:
                    if duration < min_duration_violate:
                        min_duration_violate = duration
                        chosen_route_name_violate = route_name
                        insert_idx_violate = i

        if chosen_route_name_no_violate is not None:
            actions.append(AllocateAction(
                order_id=oncall_order.id,
                route_name=chosen_route_name_no_violate,
                insert_index=route_original_indexes[chosen_route_name_no_violate][insert_idx_no_violate]
            ))
            route_plan_dict[chosen_route_name_no_violate].insert(insert_idx_no_violate, oncall_order)
            route_original_indexes[chosen_route_name_no_violate].insert(
                insert_idx_no_violate, route_original_indexes[chosen_route_name_no_violate][insert_idx_no_violate]
            )

        elif chosen_route_name_violate is not None:
            actions.append(AllocateAction(
                order_id=oncall_order.id,
                route_name=chosen_route_name_violate,
                insert_index=route_original_indexes[chosen_route_name_violate][insert_idx_violate]
            ))
            route_plan_dict[chosen_route_name_violate].insert(insert_idx_violate, oncall_order)
            route_original_indexes[chosen_route_name_violate].insert(
                insert_idx_violate, route_original_indexes[chosen_route_name_violate][insert_idx_violate]
            )

        else:
            actions.append(PostponeAction(order_id=oncall_order.id))

    actions = refresh_segment_index(actions)

    return actions


# Greedy: assign each on-call order to the closest stop on existing route.
if __name__ == "__main__":
    env = Env(
        scenario="oncall_routing", topology="example", start_tick=480, durations=960,
    )

    env.reset(keep_seed=True)
    metrics, decision_event, is_done = env.step(None)
    while not is_done:
        assert isinstance(decision_event, OncallRoutingPayload)
        print(f"Processing {len(decision_event.oncall_orders)} oncall orders at tick {env.tick}.")
        metrics, decision_event, is_done = env.step(_get_actions(env, decision_event))

    print(metrics)
