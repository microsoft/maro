# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List, Tuple

from maro.backends.frame import SnapshotList
from maro.simulator.scenarios.oncall_routing import OncallRoutingPayload, Order
from maro.simulator.scenarios.oncall_routing.duration_time_predictor import EstimatedDurationPredictor

EST_ORDER_PROCESSING_TIME = 0 # Processing time for 1 order. Used to estimate the RTB time.


def get_expected_RTB_tick(orders: List[Order], start_tick: int, predictor: EstimatedDurationPredictor) -> int:
    # The current predictor is time-independent, which is ignored here (could be used to accelerate the calculation).
    tick = start_tick
    for src, dest in zip(orders[:-1], orders[1:]):
        # TODO: simulate the order status?
        # TODO: improve the order processing time?
        tick += EST_ORDER_PROCESSING_TIME
        tick += predictor.predict(tick=tick, source_coordinate=src.coord, target_coordinate=dest.coord)
        # No processing time need for the last order -- Dummy.
    return tick

def get_expected_RTB_tick_with_oncall(
    orders: List[Order],
    start_tick: int,
    predictor: EstimatedDurationPredictor,
    oncall: Order,
    insert_index: int
) -> int:
    # The current predictor is time-independent, which is ignored here (could be used to accelerate the calculation).
    assert 0 <= insert_index < len(orders)
    orders.insert(insert_index, oncall)
    expected_RTB = get_expected_RTB_tick(orders, start_tick, predictor)
    orders.pop(insert_index)
    return expected_RTB

def get_state_info_dict(tick: int, event: OncallRoutingPayload, snapshot_list: SnapshotList) -> dict:
    """
    High-level policy features:
    - current positions of all couriers
    - current expected RTB time of each courier
    - stops on all routes as well as their time windows
    - positions of all on-call stops and their time windows
    - for each on-call stop A and route R, the earliest expected RTB of the courier
    if inserting A to R heuristically (the optimal position to insert A while respecting all time-windows)

    Low-level policy features:
    - for simplicity, low-level policy can be heuristic, namely, inserting each on-call stop to a position that
    i) does not break time windows of all stops;
    ii) minimize the total travel time of the courier
    - stops as well as their time windows on the given route
    - position and its time window for each on-call stop to be inserted
    """

    # Positions of all on-call stops and their time windows
    oncall_orders: List[Order] = event.oncall_orders

    # Stops on all routes as well as their time windows
    planned_orders: Dict[str, List[Order]] = event.route_plan_dict

    predictor: EstimatedDurationPredictor = event.estimated_duration_predictor

    # Current positions of all carriers (route name as the key)
    carriers_in_stop: List[bool] = (snapshot_list["carriers"][tick::"in_stop"] == 1).tolist()
    carrier_position_dict: Dict[str, Tuple[bool, int, int]] = {
        route_name: [
            carriers_in_stop[meta["carrier_idx"]],
            meta["estimated_duration_to_the_next_stop"],
            meta["estimated_next_departure_tick"]
        ]
        for route_name, meta in event.route_meta_info_dict.items()
    }

    # Current expected RTB of each carriers (route name as the key)
    expected_RTB: Dict[str, int] = {
        route_name: get_expected_RTB_tick(
            orders=orders,
            start_tick=tick + 0 if carrier_position_dict[route_name][0] else carrier_position_dict[route_name][1],
            predictor=predictor
        )
        for route_name, orders in planned_orders.items()
    }

    # For each on-call stop A and route R, the earliest expected RTB of the courier
    earliest_expected_RTB: Dict[str, Dict[str, int]] = {
        oncall.id: {
            route_name: min([
                get_expected_RTB_tick_with_oncall(
                    orders=orders,
                    start_tick=tick + 0 if carrier_position_dict[route_name][0] else carrier_position_dict[route_name][1],
                    predictor=predictor,
                    oncall=oncall,
                    insert_index=idx
                )
                for idx in range(1 - carrier_position_dict[route_name][0], len(orders))
            ]) if len(orders) - (1 - carrier_position_dict[route_name][0]) > 0 else -1
            for route_name, orders in planned_orders.items()
        }
        for oncall in oncall_orders
    }

    return {
        # Corresponding feature: Current positions of all carriers.
        # type: Dict[str, Tuple[bool, int, int]].
        # key: route_name.
        # value[0]: currently, whether the carrier of this route is in a stop or not.
        # value[1]: if the carrier is not in a stop, the estimated duration to arrive at the next stop.
        # value[2]: if the carrier is in a stop, the estimated tick for the carrier to depart from current stop.
        "carrier_position_info": carrier_position_dict,
        # Corresponding feature: Current expected RTB of each carriers.
        # type: Dict[str, int]
        # key: route_name
        # value: the expected RTB time, unit: env tick.
        "expected_RTB_time": expected_RTB,
        # Corresponding feature: stops on all routes as well as their time windows
        # type: Dict[str, List[Order]]
        # key: route_name
        # value: the planned order list. The carrier will process the orders in the order they are in the list.
        "orders_in_plan": planned_orders,
        # Corresponding feature: Positions of all on-call stops and their time windows
        # type: List[Order]
        "oncall_orders": oncall_orders,
        # Corresponding feature: For each on-call stop A and route R, the earliest expected RTB of the courier
        # type: Dict[str, Dict[str, int]]
        # key-1: the on-call order id.
        # key-2: route_name
        # value: the earlist expected RTB time if insert this order to this route
        "expected_RTB_time_with_oncall": earliest_expected_RTB
    }


if __name__ == "__main__":
    from datetime import datetime
    from maro.simulator import Env

    env = Env(
        scenario="oncall_routing", topology="example", start_tick=0, durations=400,
    )

    env.reset(keep_seed=True)
    metrics, decision_event, is_done = env.step(None)
    while not is_done:
        assert isinstance(decision_event, OncallRoutingPayload)

        start_time = datetime.now()
        state_info = get_state_info_dict(env.tick, decision_event, env.snapshot_list)
        end_time = datetime.now()
        duration = (end_time - start_time).microseconds // 1000
        print(
            f"Tick {env.tick}: "
            f"{len(decision_event.oncall_orders)} on-call orders, "
            f"{len(decision_event.route_plan_dict)} routes, "
            f"time spent for state shaping: {duration} ms"
        )

        metrics, decision_event, is_done = env.step(None)

    print(state_info)
    print(metrics)
