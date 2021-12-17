# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List, Tuple

from maro.backends.frame import SnapshotList
from maro.simulator.scenarios.oncall_routing import OncallRoutingPayload, Order
from maro.simulator.scenarios.oncall_routing.duration_time_predictor import EstimatedDurationPredictor

EST_ORDER_PROCESSING_TIME = 0 # Processing time for 1 order. Used to estimate the RTB time.

USE_TIME_INDEPENDENT_PREDICTOR = True


def get_time_independent_RTB_with_and_without_oncall(
    tick: int,
    planned_orders: Dict[str, List[Order]],
    oncall_orders: List[Order],
    carrier_position_dict: Dict[str, Tuple[bool, int, int]],
    predictor: EstimatedDurationPredictor
):
    # Expected arrival time of each stop without inserting any on-call order.
    expected_arrival_time_without_oncall: Dict[str, List[int]] = {}
    for route_name, orders in planned_orders.items():
        if len(orders) <= 0:
            continue
        cur_tick = tick + 0 if carrier_position_dict[route_name][0] else carrier_position_dict[route_name][1]
        expected_arrival_time_without_oncall[route_name] = []
        for src, dest in zip(orders[:-1], orders[1:]):
            expected_arrival_time_without_oncall[route_name].append(cur_tick)
            cur_tick += EST_ORDER_PROCESSING_TIME
            cur_tick += predictor.predict(tick=None, source_coordinate=src.coord, target_coordinate=dest.coord)
        expected_arrival_time_without_oncall[route_name].append(cur_tick)

    # The maximum postpone duration for each stop that do no violate the time window constraint.
    maximum_postpone_duration: Dict[str, List[int]] = {}
    for route_name, orders in planned_orders.items():
        if len(orders) <= 0:
            continue
        maximum_postpone_duration[route_name] = [-1] * len(orders)
        maximum_postpone_duration[route_name][-1] = (
            orders[-1].close_time - expected_arrival_time_without_oncall[route_name][-1]
        )
        idx = len(orders) - 2
        while idx >= 0:
            maximum_postpone_duration[route_name][idx] = min(
                maximum_postpone_duration[route_name][idx + 1],
                orders[idx].close_time - expected_arrival_time_without_oncall[route_name][idx]
            )
            idx -= 1

    # The minimum delay with and without violation.
    minimum_delay_without_violation: Dict[str, Dict[str, int]] = {}
    minimum_delay_with_violation: Dict[str, Dict[str, int]] = {}
    for oncall in oncall_orders:
        minimum_delay_without_violation[oncall.id] = {}
        minimum_delay_with_violation[oncall.id] = {}
        for route_name, orders in planned_orders.items():
            if len(orders) <= 0:
                continue
            minimum_delay_without_violation[oncall.id][route_name] = float("inf")
            minimum_delay_with_violation[oncall.id][route_name] = float("inf")
            if carrier_position_dict[route_name][0]:
                delay = (
                    predictor.predict(
                        tick=None,
                        source_coordinate=carrier_position_dict[route_name][3],
                        target_coordinate=oncall.coord
                    )
                    + predictor.predict(
                        tick=None,
                        source_coordinate=oncall.coord,
                        target_coordinate=orders[0].coord
                    )
                    - predictor.predict(
                        tick=None,
                        source_coordinate=carrier_position_dict[route_name][3],
                        target_coordinate=orders[0].coord
                    )
                )
                if delay <= maximum_postpone_duration[route_name][0]:
                    minimum_delay_without_violation[oncall.id][route_name] = min(
                        minimum_delay_without_violation[oncall.id][route_name],
                        delay
                    )
                else:
                    minimum_delay_with_violation[oncall.id][route_name] = min(
                        minimum_delay_with_violation[oncall.id][route_name],
                        delay
                    )
            for idx in range(1, len(orders)):
                delay = (
                    predictor.predict(
                        tick=None,
                        source_coordinate=orders[idx - 1].coord,
                        target_coordinate=oncall.coord
                    )
                    + predictor.predict(
                        tick=None,
                        source_coordinate=oncall.coord,
                        target_coordinate=orders[idx].coord
                    )
                    - predictor.predict(
                        tick=None,
                        source_coordinate=orders[idx - 1].coord,
                        target_coordinate=orders[idx].coord
                    )
                )
                if delay <= maximum_postpone_duration[route_name][idx]:
                    minimum_delay_without_violation[oncall.id][route_name] = min(
                        minimum_delay_without_violation[oncall.id][route_name],
                        delay
                    )
                else:
                    minimum_delay_with_violation[oncall.id][route_name] = min(
                        minimum_delay_with_violation[oncall.id][route_name],
                        delay
                    )

    # The return values.
    expected_RTB_without_oncall: Dict[str, int] = {
        route_name: expected_arrival_time_without_oncall[route_name][-1] if len(orders) > 0 else -1
        for route_name, orders in planned_orders.items()
    }

    earliest_expected_RTB_with_oncall: Dict[str, Dict[str, int]] = {
        oncall.id: {
            expected_arrival_time_without_oncall[route_name][-1] + (
                minimum_delay_without_violation[oncall.id][route_name]
                if minimum_delay_without_violation[oncall.id][route_name] < maximum_postpone_duration[route_name][-1]
                else minimum_delay_with_violation[oncall.id][route_name]
            ) if len(orders) > 0 else -1
            for route_name, orders in planned_orders.items()
        }
        for oncall in oncall_orders
    }

    return expected_RTB_without_oncall, earliest_expected_RTB_with_oncall

def get_expected_RTB_tick(orders: List[Order], start_tick: int, predictor: EstimatedDurationPredictor) -> int:
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

def get_time_dependent_RTB_with_and_without_oncall(
    tick: int,
    planned_orders: Dict[str, List[Order]],
    oncall_orders: List[Order],
    carrier_position_dict: Dict[str, Tuple[bool, int, int]],
    predictor: EstimatedDurationPredictor
):
    # TODO: add the time window check.
    # Current expected RTB of each carriers (route name as the key)
    expected_RTB_without_oncall: Dict[str, int] = {
        route_name: get_expected_RTB_tick(
            orders=orders,
            start_tick=tick + 0 if carrier_position_dict[route_name][0] else carrier_position_dict[route_name][1],
            predictor=predictor
        )
        for route_name, orders in planned_orders.items()
    }

    # For each on-call stop A and route R, the earliest expected RTB of the courier
    earliest_expected_RTB_with_oncall: Dict[str, Dict[str, int]] = {
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

    return expected_RTB_without_oncall, earliest_expected_RTB_with_oncall

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
            meta["estimated_next_departure_tick"],
            meta["current_staying_stop_coordinate"]
        ]
        for route_name, meta in event.route_meta_info_dict.items()
    }

    # Current expected RTB of each carriers (route name as the key) with and without oncall.
    if USE_TIME_INDEPENDENT_PREDICTOR:
        (
            expected_RTB_without_oncall, earliest_expected_RTB_with_oncall
        ) = get_time_independent_RTB_with_and_without_oncall(
            tick, planned_orders, oncall_orders, carrier_position_dict, predictor
        )
    else:
        expected_RTB_without_oncall, earliest_expected_RTB_with_oncall = get_time_dependent_RTB_with_and_without_oncall(
            tick, planned_orders, oncall_orders, carrier_position_dict, predictor
        )

    return {
        # Corresponding feature: Current positions of all carriers.
        # type: Dict[str, Tuple[bool, int, int]].
        # key: route_name.
        # value[0]: currently, whether the carrier of this route is in a stop or not.
        # value[1]: if the carrier is not in a stop, the estimated duration to arrive at the next stop.
        # value[2]: if the carrier is in a stop, the estimated tick for the carrier to depart from current stop.
        # value[3]: if the carrier is in a stop, the coordinate the carrier is in.
        "carrier_position_info": carrier_position_dict,
        # Corresponding feature: Current expected RTB of each carriers.
        # type: Dict[str, int]
        # key: route_name
        # value: the expected RTB time, unit: env tick.
        "expected_RTB_time_without_oncall": expected_RTB_without_oncall,
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
        "expected_RTB_time_with_oncall": earliest_expected_RTB_with_oncall
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
        duration = (end_time - start_time).total_seconds()
        print(
            f"Tick {env.tick}: "
            f"{len(decision_event.oncall_orders)} on-call orders, "
            f"{len(decision_event.route_plan_dict)} routes, "
            f"time spent for state shaping: {duration} s"
        )

        metrics, decision_event, is_done = env.step(None)

    # print(state_info)
    print(metrics)
