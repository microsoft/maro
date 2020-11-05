# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from math import ceil
from typing import List

from yaml import safe_load

from maro.simulator.utils import seed
from maro.utils.exception.data_lib_exeption import CimGeneratorInvalidParkingDuration

from .entities import CimDataCollection, OrderGenerateMode, Stop
from .global_order_proportion import GlobalOrderProportion
from .port_parser import PortsParser
from .route_parser import RoutesParser
from .utils import apply_noise, route_init_rand
from .vessel_parser import VesselsParser

CIM_GENERATOR_VERSION = 0x000001


class CimDataGenerator:
    """Utility to generate cim data from configuration file."""

    def __init__(self):
        # parsers
        self._ports_parser = PortsParser()
        self._vessels_parser = VesselsParser()
        self._routes_parser = RoutesParser()
        self._global_order_proportion = GlobalOrderProportion()

    def gen_data(self, config_file: str, max_tick: int, start_tick: int = 0) -> CimDataCollection:
        """Generate data with specified configurations.

        Args:
            config_file(str): File of configuration (yaml).
            max_tick(int): Max tick to generate.
            start_tick(int): Start tick to generate.

        Returns:
            CimDataCollection: Data collection contains all cim data.
        """

        # read config
        with open(config_file, "r") as fp:
            conf: dict = safe_load(fp)

        topology_seed = conf["seed"]

        # set seed to generate data
        seed(topology_seed)

        # misc configurations
        total_containers = conf["total_containers"]
        past_stop_number, future_stop_number = conf["stop_number"]
        cntr_volumes = conf["container_volumes"]
        order_mode = OrderGenerateMode(conf["order_generate_mode"])

        # parse configurations
        vessel_mapping, vessels_setting = self._vessels_parser.parse(conf["vessels"])
        port_mapping, ports_setting = self._ports_parser.parse(conf["ports"], total_containers)
        route_mapping, routes = self._routes_parser.parse(conf["routes"])
        global_order_proportion = self._global_order_proportion.parse(
            conf["container_usage_proportion"],
            total_containers, start_tick=start_tick, max_tick=max_tick)

        # extend routes with specified tick range
        vessels_stops, vessel_period_no_noise = self._extend_route(
            future_stop_number, max_tick, vessels_setting, ports_setting, port_mapping, routes, route_mapping)

        return CimDataCollection(
            total_containers,
            past_stop_number,
            future_stop_number,
            cntr_volumes[0],
            order_mode,
            ports_setting,
            port_mapping,
            vessels_setting,
            vessel_mapping,
            vessels_stops,
            global_order_proportion,
            routes,
            route_mapping,
            vessel_period_no_noise,
            max_tick,
            topology_seed,
            CIM_GENERATOR_VERSION)

    def _extend_route(
        self, future_stop_number: int, max_tick: int,
        vessels_setting, ports_setting, port_mapping, routes, route_mapping
    ):
        """Extend route with specified tick range."""

        vessels_stops: List[List[Stop]] = []
        vessel_period_no_noise: list = []

        # fill the result stops with empty list
        # NOTE: not using [[]] * N
        for _ in range(len(vessels_setting)):
            vessels_stops.append([])

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
                parking_duration = ceil(apply_noise(duration, duration_noise, route_init_rand))

                if parking_duration <= 0:
                    raise CimGeneratorInvalidParkingDuration()

                # a new stop
                stop = Stop(stop_index,
                            tick,
                            tick + parking_duration,
                            port_idx,
                            vessel_setting.index)

                # append to current vessels stops list
                vessels_stops[vessel_setting.index].append(stop)

                # use distance and speed (all with noise) to calculate tick of arrival and departure
                distance = cur_route_point.distance

                # apply noise to speed
                noised_speed = apply_noise(speed, speed_noise, route_init_rand)
                sailing_duration = ceil(distance / noised_speed)

                # next tick
                tick += parking_duration + sailing_duration

                # sailing durations without noise
                whole_duration_no_noise = duration + ceil(distance / speed)

                # only add period at 1st route circle
                period_no_noise += (whole_duration_no_noise if len(
                    vessels_stops[vessel_setting.index]) <= route_length else 0)

                # next location index
                loc_idx_in_route = (loc_idx_in_route + 1) % route_length

                # counter to append extra stops which after max tick for future predict
                extra_stop_counter += (1 if tick > max_tick else 0)

                stop_index += 1

            vessel_period_no_noise.append(period_no_noise)

        return vessels_stops, vessel_period_no_noise
