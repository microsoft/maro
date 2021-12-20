# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List, Tuple

from maro.backends.frame import SnapshotList
from maro.simulator.scenarios.oncall_routing import Coordinate, OncallRoutingPayload, Order
from maro.simulator.scenarios.oncall_routing.duration_time_predictor import EstimatedDurationPredictor

EST_ORDER_PROCESSING_TIME = 0  # Processing time for 1 order. Used to estimate the RTB time.

USE_TIME_INDEPENDENT_PREDICTOR = True


def _get_insertion_delay(
    predictor: EstimatedDurationPredictor,
    src: Coordinate,
    tgt: Coordinate,
    oncall_order: Coordinate
) -> int:
    ret = 0
    ret += predictor.predict(-1, src, oncall_order)
    ret += predictor.predict(-1, oncall_order, tgt)
    ret -= predictor.predict(-1, src, tgt)
    return ret


def _get_expected_RTB_tick(orders: List[Order], start_tick: int, predictor: EstimatedDurationPredictor) -> int:
    tick = start_tick
    for src, dest in zip(orders[:-1], orders[1:]):
        # TODO: simulate the order status?
        # TODO: improve the order processing time?
        tick += EST_ORDER_PROCESSING_TIME
        tick += predictor.predict(tick=tick, source_coordinate=src.coord, target_coordinate=dest.coord)
        # No processing time need for the last order -- Dummy.
    return tick


def _get_expected_RTB_tick_with_oncall(
    orders: List[Order],
    start_tick: int,
    predictor: EstimatedDurationPredictor,
    oncall: Order,
    insert_index: int
) -> int:
    # The current predictor is time-independent, which is ignored here (could be used to accelerate the calculation).
    assert 0 <= insert_index < len(orders)
    orders.insert(insert_index, oncall)
    expected_RTB = _get_expected_RTB_tick(orders, start_tick, predictor)
    orders.pop(insert_index)
    return expected_RTB


def _get_time_independent_RTB_with_and_without_oncall(
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

        estimated_duration_to_the_next_stop = carrier_position_dict[route_name][1]
        cur_tick = tick + estimated_duration_to_the_next_stop

        expected_arrival_time_without_oncall[route_name] = []
        for src, dest in zip(orders[:-1], orders[1:]):
            expected_arrival_time_without_oncall[route_name].append(cur_tick)
            cur_tick += EST_ORDER_PROCESSING_TIME
            cur_tick += predictor.predict(tick=-1, source_coordinate=src.coord, target_coordinate=dest.coord)
        expected_arrival_time_without_oncall[route_name].append(cur_tick)

    # The max postpone duration for each stop that do no violate the time window constraint.
    max_postpone_duration: Dict[str, List[int]] = {}
    for route_name, orders in planned_orders.items():
        if len(orders) <= 0:
            continue

        max_postpone_duration[route_name] = [
            order.close_time - arr_time
            for order, arr_time in zip(orders, expected_arrival_time_without_oncall[route_name])
        ]
        for i in range(len(max_postpone_duration[route_name]) - 1)[::-1]:
            max_postpone_duration[route_name][i] = min(
                max_postpone_duration[route_name][i],
                max_postpone_duration[route_name][i + 1]
            )

    # The min delay with and without violation.
    min_delay_no_violate: Dict[str, Dict[str, float]] = {}
    min_delay_violate: Dict[str, Dict[str, float]] = {}
    for oncall in oncall_orders:
        no_violate: Dict[str, float] = {}
        violate: Dict[str, float] = {}
        for route_name, orders in planned_orders.items():
            if len(orders) <= 0:
                continue

            no_violate[route_name] = violate[route_name] = float("inf")
            carrier_in_stop = carrier_position_dict[route_name][0]
            current_staying_stop_coordinate = carrier_position_dict[route_name][3]
            for i in range(len(orders)):
                if i == 0 and not carrier_in_stop:
                    continue
                last_coord = current_staying_stop_coordinate if i == 0 else orders[i - 1].coord
                delay = _get_insertion_delay(predictor, last_coord, orders[i].coord, oncall.coord)
                if delay <= max_postpone_duration[route_name][i]:
                    no_violate[route_name] = min(no_violate[route_name], delay)
                else:
                    violate[route_name] = min(violate[route_name], delay)
        min_delay_no_violate[oncall.id] = no_violate
        min_delay_violate[oncall.id] = violate

    # The return values.
    expected_RTB_without_oncall: Dict[str, float] = {
        route_name: expected_arrival_time_without_oncall[route_name][-1] if len(orders) > 0 else -1
        for route_name, orders in planned_orders.items()
    }

    earliest_expected_RTB_with_oncall: Dict[str, Dict[str, float]] = {
        oncall.id: {
            route_name: expected_arrival_time_without_oncall[route_name][-1] + (
                min_delay_no_violate[oncall.id][route_name]
                if min_delay_no_violate[oncall.id][route_name] < max_postpone_duration[route_name][-1]
                else min_delay_violate[oncall.id][route_name]
            ) if len(orders) > 0 else -1
            for route_name, orders in planned_orders.items()
        }
        for oncall in oncall_orders
    }

    return expected_RTB_without_oncall, earliest_expected_RTB_with_oncall


def _get_time_dependent_RTB_with_and_without_oncall(
    tick: int,
    planned_orders: Dict[str, List[Order]],
    oncall_orders: List[Order],
    carrier_position_dict: Dict[str, Tuple[bool, int, int]],
    predictor: EstimatedDurationPredictor
):
    # TODO: add the time window check.
    # Current expected RTB of each carriers (route name as the key)
    expected_RTB_without_oncall: Dict[str, int] = {
        route_name: _get_expected_RTB_tick(
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
                _get_expected_RTB_tick_with_oncall(
                    orders=orders,
                    start_tick=tick + 0 if carrier_position_dict[route_name][0]
                    else carrier_position_dict[route_name][1],
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


def get_RTB_with_and_without_oncall(
    tick: int,
    planned_orders: Dict[str, List[Order]],
    oncall_orders: List[Order],
    carrier_position_dict: Dict[str, Tuple[bool, int, int]],
    predictor: EstimatedDurationPredictor,
    time_dependent: bool
):
    func = _get_time_dependent_RTB_with_and_without_oncall if time_dependent \
        else _get_time_independent_RTB_with_and_without_oncall
    return func(tick, planned_orders, oncall_orders, carrier_position_dict, predictor)


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
    expected_RTB_without_oncall, earliest_expected_RTB_with_oncall = get_RTB_with_and_without_oncall(
        tick, planned_orders, oncall_orders, carrier_position_dict, predictor, not USE_TIME_INDEPENDENT_PREDICTOR
    )

    """
    Returned dict:
        "carrier_position_info"
            Corresponding feature: Current positions of all carriers.
            type: Dict[str, Tuple[bool, int, int]].
            key: route_name.
            value[0]: currently, whether the carrier of this route is in a stop or not.
            value[1]: if the carrier is not in a stop, the estimated duration to arrive at the next stop.
            value[2]: if the carrier is in a stop, the estimated tick for the carrier to depart from current stop.
            value[3]: if the carrier is in a stop, the coordinate the carrier is in.

        "expected_RTB_time_without_oncall"
            Corresponding feature: Current expected RTB of each carriers.
            type: Dict[str, int]
            key: route_name
            value: the expected RTB time, unit: env tick.

        "orders_in_plan"
            Corresponding feature: stops on all routes as well as their time windows
            type: Dict[str, List[Order]]
            key: route_name
            value: the planned order list. The carrier will process the orders in the order they are in the list.

        "oncall_orders"
            Corresponding feature: Positions of all on-call stops and their time windows
            type: List[Order]

        "expected_RTB_time_with_oncall"
            Corresponding feature: For each on-call stop A and route R, the earliest expected RTB of the courier
            type: Dict[str, Dict[str, int]]
            key-1: the on-call order id.
            key-2: route_name
            value: the earliest expected RTB time if insert this order to this route
    """
    return {
        "carrier_position_info": carrier_position_dict,
        "expected_RTB_time_without_oncall": expected_RTB_without_oncall,
        "orders_in_plan": planned_orders,
        "oncall_orders": oncall_orders,
        "expected_RTB_time_with_oncall": earliest_expected_RTB_with_oncall,
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
