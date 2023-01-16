# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from math import ceil
from typing import List

from yaml import safe_load

from maro.simulator.utils import random, seed

from .entities import CimSyntheticDataCollection, OrderGenerateMode, Stop
from .parsers import parse_global_order_proportion, parse_ports, parse_routes, parse_vessels
from .utils import ROUTE_INIT_RAND_KEY, apply_noise

CIM_GENERATOR_VERSION = 0x000001


def _extend_route(
    future_stop_number: int,
    max_tick: int,
    vessels_setting,
    port_mapping,
    routes,
    route_mapping,
) -> (List[List[Stop]], List[int]):
    """Extend route with specified tick range."""

    vessel_stops: List[List[Stop]] = []
    vessel_period_without_noise: List[int] = []

    # fill the result stops with empty list
    # NOTE: not using [[]] * N
    for _ in range(len(vessels_setting)):
        vessel_stops.append([])

    # extend for each vessel
    for vessel_setting in vessels_setting:
        route_name = vessel_setting.route_name

        # route definition points from configuration
        route_points = routes[route_mapping[route_name]]
        route_length = len(route_points)

        loc_idx_in_route = 0

        # find the start point
        while route_points[loc_idx_in_route].port_name != vessel_setting.start_port_name:
            loc_idx_in_route += 1

        # update initial value fields
        speed = vessel_setting.sailing_speed
        speed_noise = vessel_setting.sailing_noise
        duration = vessel_setting.parking_duration
        duration_noise = vessel_setting.parking_noise

        tick = 0
        period_no_noise = 0
        extra_stop_counter = 0
        stop_index = 0

        # unpack the route by max tick and future stop number
        while extra_stop_counter <= future_stop_number:
            cur_route_point = route_points[loc_idx_in_route]
            port_idx = port_mapping[cur_route_point.port_name]

            # apply noise to parking duration
            parking_duration = ceil(apply_noise(duration, duration_noise, random[ROUTE_INIT_RAND_KEY]))
            assert parking_duration > 0

            # a new stop
            stop = Stop(
                stop_index,
                tick,
                tick + parking_duration,
                port_idx,
                vessel_setting.index,
            )

            # append to current vessels stops list
            vessel_stops[vessel_setting.index].append(stop)

            # use distance_to_next_port and speed (all with noise) to calculate tick of arrival and departure
            distance_to_next_port = cur_route_point.distance_to_next_port

            # apply noise to speed
            noised_speed = apply_noise(speed, speed_noise, random[ROUTE_INIT_RAND_KEY])
            sailing_duration = ceil(distance_to_next_port / noised_speed)

            # next tick
            tick += parking_duration + sailing_duration

            # sailing durations without noise
            whole_duration_no_noise = duration + ceil(distance_to_next_port / speed)

            # only add period at 1st route circle
            period_no_noise += (
                whole_duration_no_noise
                if len(
                    vessel_stops[vessel_setting.index],
                )
                <= route_length
                else 0
            )

            # next location index
            loc_idx_in_route = (loc_idx_in_route + 1) % route_length

            # counter to append extra stops which after max tick for future predict
            extra_stop_counter += 1 if tick > max_tick else 0

            stop_index += 1

        vessel_period_without_noise.append(period_no_noise)

    return vessel_stops, vessel_period_without_noise


def gen_cim_data(
    config_file: str,
    max_tick: int,
    start_tick: int = 0,
    topology_seed: int = None,
) -> CimSyntheticDataCollection:
    """Generate data with specified configurations.

    Args:
        config_file(str): File of configuration (yaml).
        max_tick(int): Max tick to generate.
        start_tick(int): Start tick to generate.
        topology_seed(int): Random seed of the business engine. \
            'None' means using the seed in the configuration file.

    Returns:
        CimSyntheticDataCollection: Data collection contains all cim data.
    """

    # read config
    with open(config_file, "r") as fp:
        conf: dict = safe_load(fp)

    if topology_seed is None:
        topology_seed = conf["seed"]

    # set seed to generate data
    seed(topology_seed)

    # misc configurations
    total_containers = conf["total_containers"]
    past_stop_number, future_stop_number = conf["stop_number"]
    container_volumes = conf["container_volumes"]

    # parse configurations
    vessel_mapping, vessels_setting = parse_vessels(conf["vessels"])
    port_mapping, ports_setting = parse_ports(conf["ports"], total_containers)
    route_mapping, routes = parse_routes(conf["routes"])
    global_order_proportion = parse_global_order_proportion(
        conf["container_usage_proportion"],
        total_containers,
        start_tick=start_tick,
        max_tick=max_tick,
    )

    # extend routes with specified tick range
    vessel_stops, vessel_period_without_noise = _extend_route(
        future_stop_number,
        max_tick,
        vessels_setting,
        port_mapping,
        routes,
        route_mapping,
    )

    return CimSyntheticDataCollection(
        # Port
        port_settings=ports_setting,
        port_mapping=port_mapping,
        # Vessel
        vessel_settings=vessels_setting,
        vessel_mapping=vessel_mapping,
        # Stop
        vessel_stops=vessel_stops,
        # Route
        routes=routes,
        route_mapping=route_mapping,
        # Vessel Period
        vessel_period_without_noise=vessel_period_without_noise,
        # Volume/Container
        container_volume=container_volumes[0],
        # Cost Factors
        load_cost_factor=conf["load_cost_factor"],
        dsch_cost_factor=conf["dsch_cost_factor"],
        # Visible Voyage Window
        past_stop_number=past_stop_number,
        future_stop_number=future_stop_number,
        # Time Length of the Data Collection
        max_tick=max_tick,
        # Random Seed for Data Generation
        seed=topology_seed,
        # For Order Generation
        total_containers=total_containers,
        order_mode=OrderGenerateMode(conf["order_generate_mode"]),
        order_proportion=global_order_proportion,
        # Data Generator Version
        version=str(CIM_GENERATOR_VERSION),
    )
