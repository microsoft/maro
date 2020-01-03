# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from enum import Enum
from math import ceil, floor

from maro.simulator.utils.random import random

import numpy as np
from yaml import safe_load

from maro.simulator.scenarios.ecr.common import Stop, Order


# we need 4 random sequence to extend the route info, order distribution, buffer time and order number generating
route_init_rand = random["route_init"]
order_init_rand = random["order_init"]
buffer_tick_rand = random["buffer_time"]
order_num_rand = random["order_number"]


def sum_normalize(l):
    t = sum(l)

    return [d / t for d in l]


def apply_noise(value, noise, rand):
    v = rand.uniform(-noise, noise)
    return value + v


class OrderGenerateMode(Enum):
    FIXED = "fixed"
    UNFIXED = "unfixed"


class VesselDurationWrong(Exception):
    """
    Vessel parking duration is not right
    """

    def __init__(self, msg):
        super().__init__(msg)


class GlobalOrderProportion:
    """
    Proportion of order

    This class will extend (interpolate) the order proportion from config
    """
    def __init__(self, proportion_conf, max_tick: int, total_container: int):
        self.period = proportion_conf["period"]
        self._sample_nodes = [(x, y) for x, y in proportion_conf["sample_nodes"]]
        self._noise = proportion_conf["sample_noise"]
        self._max_tick = max_tick
        self._total_container = total_container

        self.order_period_distribution = None
        self.order_distribution = [0] * max_tick
        self.total_order = 0  # total order after unpack to max tick

        self._unpack_to_period()
        self._unpack_to_max_tick()

    def _unpack_to_period(self):
        # check if there is 0 and max_tick - 1 node exist
        if self._sample_nodes[0][0] != 0:
            self._sample_nodes.insert(0, (0, 0))

        if self._sample_nodes[-1][0] != self.period - 1:
            self._sample_nodes.append((self.period - 1, 0))

        # our xp is period
        xp = [node[0] for node in self._sample_nodes]
        yp = [node[1] for node in self._sample_nodes]

        self.order_period_distribution = np.interp([t for t in range(self.period)], xp, yp)

    def _unpack_to_max_tick(self):
        for t in range(self._max_tick):
            self.order_distribution[t] = self.order_period_distribution[t % self.period]

            # apply noise if the distribution not zero
            if self.order_distribution[t] != 0:
                self.order_distribution[t] = apply_noise(self.order_distribution[t], self._noise, order_init_rand)
                # clip and gen order
                self.order_distribution[t] = floor(max(0, min(1, self.order_distribution[t])) * self._total_container)

        # update total order number
        self.total_order = sum(self.order_distribution)


class PortsInfo:
    """
    Information parsed from config
    """
    def __init__(self, port_conf, total_containers: int):
        self._conf = port_conf
        self._total_containers = total_containers

        # if we should keep the random sequence to make sure the order sequence will not be affected
        self._length = 0
        self.mapping = {}  # name->idx
        self._empty_returns = []
        self._full_returns = []
        self._order_distributions = []  # list of tuple ((source ratio, noise), [(target_id, ratio, noise), ])
        self._initial_values = {"empty": [], "capacity": []}

        self._parse()

    def __len__(self):
        return self._length

    @property
    def initial_value(self):
        """
        Initial values of port include "empty" and "capacity"

        Returns:
            dictionary with "empty" and "capacity" as key, value is list of value (index is the port idx)
        """
        return self._initial_values

    def get_order_distribution(self, port_idx: int):
        """
        Get order distribution as configured

        Args:
            port_idx (int): port index to get

        Returns:
            list of tuple ((source ratio, noise), [(target_id, ratio, noise), ])
        """
        return self._order_distributions[port_idx]

    def get_full_return_buffer_tick(self, port_idx: int):
        """
        Get the buffer time for full return (with noise)

        Args:
            port_idx (int): port index to get

        Returns:
            buffer time (int)
        """
        ticks, noise = self._full_returns[port_idx]

        # NOTE: here we keep the random state used to restore it after apply noise,
        # so that it will not affect order genration

        return ceil(apply_noise(ticks, noise, buffer_tick_rand))

    def get_empty_return_buffer_tick(self, port_idx: int):
        """
        Get the buffer time for empty return (with noise)

        Args:
            port_idx (int): port index to get

        Returns:
            buffer time (int)
        """
        ticks, noise = self._empty_returns[port_idx]

        return ceil(apply_noise(ticks, noise, buffer_tick_rand))

    def _parse(self):
        # sum of ratio cannot large than 1
        total_ratio = sum([p["initial_container_proportion"] for p in self._conf.values()])
        assert round(total_ratio, 7) == 1

        # create mapping first
        for port_name, _ in self._conf.items():
            self.mapping[port_name] = self._length
            self._length += 1

        acc_empty = 0
        for port_name, port_info in self._conf.items():
            # update initial values
            empty_number = floor(port_info["initial_container_proportion"] * self._total_containers)
            acc_empty += empty_number

            self._initial_values["empty"].append(empty_number)
            self._initial_values["capacity"].append(port_info["capacity"])

            #
            full_return_conf = port_info["full_return"]
            self._full_returns.append((full_return_conf["buffer_ticks"], full_return_conf["noise"]))

            #
            empty_return_conf = port_info["empty_return"]
            self._empty_returns.append((empty_return_conf["buffer_ticks"], empty_return_conf["noise"]))

            #
            dist_conf = port_info["order_distribution"]
            source_dist_conf = dist_conf["source"]

            source_dist = (source_dist_conf["proportion"], source_dist_conf["noise"])

            targets_dist = []

            # orders distribution to destination
            if "targets" in dist_conf:
                for target_port_name, target_conf in dist_conf["targets"].items():
                    dist = (self.mapping[target_port_name], target_conf["proportion"], target_conf["noise"])

                    targets_dist.append(dist)

            self._order_distributions.append((source_dist, targets_dist))

        if acc_empty != self._total_containers:
            self._initial_values["empty"][-1] += self._total_containers - acc_empty


class RoutesInfo:
    """
    Route information parsed from config
    """
    def __init__(self, route_conf, port_mapping: dict):
        self._route_conf = route_conf
        self._routes = []  # parsed route points, list of tuple (port_idx, distance)
        self.length_dict = {}  # length of each route, key is the route index
        self.mapping = {}  # name->idx
        self._parse(port_mapping)

    def get_route_points_by_name(self, route_name: str):
        """
        Get a point at specified route name

        Args:
            route_name (str): route name

        Returns:
            list of tuple (port_idx, distance)
        """
        return self._routes[self.mapping[route_name]]

    def get_route_points_by_idx(self, route_idx: int):
        """
        Get a point at specified route index

        Args:
            route_idx (int): route index

        Returns:
            list of tuple (port_idx, distance)
        """
        return self._routes[route_idx]

    def _parse(self, port_mapping: dict):
        idx = 0

        for route_name, route_node in self._route_conf.items():
            self.mapping[route_name] = idx

            self._routes.append([(port_mapping[r["port_name"]], r["distance"]) for r in route_node])
            self.length_dict[idx] = len(self._routes[-1])

            idx += 1


class VesselsInfo:
    """
    Vessel information parsed from config
    """
    def __init__(self, stop_number, vessel_conf, port_mapping: dict, route_info: RoutesInfo, max_tick: int):
        self._max_tick = max_tick
        self._stop_numbers = stop_number
        self._vessel_conf = vessel_conf
        self._length = 0
        self._initial_values = {"capacity": [], "route": [], "period": []}
        self._route_info = route_info
        self.mapping = {}  # name->idx
        self.unpacked_routes = []  # list of unpack stops to the end tick with noise

        self._parse(route_info, port_mapping)

    def get_stops_list(self, vessel_idx: int, last_loc_idx: int, next_loc_idx: int):
        """
        Get past and future stop list

        Args:
            vessel_idx (int): vessel index
            last_loc_idx (int): last location index of vessel
            next_loc_idx (int): next location index of vessel

        Returns:
            tuple of ([past stops], [future stops])
        """
        past_stop_num = self._stop_numbers[0]
        future_stop_num = self._stop_numbers[1]

        # ignore current port if parking
        start = next_loc_idx + (1 if last_loc_idx == next_loc_idx else 0)

        if last_loc_idx == next_loc_idx:
            future_stop_list = self.predict_future_stops_from_loc(vessel_idx, start, future_stop_num)
        else:
            future_stop_list = self.predict_future_stops_from_loc(vessel_idx, start - 1, future_stop_num)

        start = max(next_loc_idx - past_stop_num, 0)

        past_stop_list = self.unpacked_routes[vessel_idx][start: next_loc_idx]

        # padding with None
        for _ in range(past_stop_num - len(past_stop_list)):
            past_stop_list.insert(0, None)

        #  TODO: remove later
        assert len(past_stop_list) + len(future_stop_list) == sum(self._stop_numbers)

        return past_stop_list, future_stop_list

    def predict_future_stops_from_loc(self, vessel_idx: int, last_loc_idx: int, stop_num: int):
        """
        Predict future stops by last location index

        Args:
            vessel_idx (int): vessel index
            last_loc_idx (int): last location index
            stop_num (int): how many stops need to predict

        Returns:
            list of Stop object
        """
        last_stop: Stop = self.unpacked_routes[vessel_idx][last_loc_idx]
        last_port_idx = last_stop.port_idx
        last_port_arrive_tick = last_stop.arrive_tick
        return self._predict_future_stops(vessel_idx, last_port_idx, last_port_arrive_tick, stop_num)

    @property
    def initial_values(self):
        """
        Initial value of vessel

        Returns:
            dictionary with "route" and "capacity" as key, value is list of value (index is vessel index)
        """
        return self._initial_values

    def get_stop(self, vessel_idx: int, loc_idx: int):
        """
        Get a stop of specified vessel and location index

        Args:
            vessel_idx (int): index of vessel
            loc_idx (int): index of location (stop)

        Returns:
            A Stop object
        """
        return self.unpacked_routes[vessel_idx][loc_idx]

    def __len__(self):
        return self._length

    def _predict_future_stops(self, vessel_idx: int, last_port_idx: int, last_port_arrive_tick: int, stop_num: int):
        """
        Do predict future stops
        """
        vessel_node = list(self._vessel_conf.values())[vessel_idx]
        speed = vessel_node["sailing"]["speed"]
        duration = vessel_node["parking"]["duration"]
        route_name = vessel_node["route"]["route_name"]
        route_points = self._route_info.get_route_points_by_name(route_name)
        route_length = len(route_points)

        last_loc_idx = -1
        for loc_idx, route_point in enumerate(route_points):
            if route_point[0] == last_port_idx:
                last_loc_idx = loc_idx
                break
        if last_loc_idx < 0:
            return []

        predicted_future_stops = []
        arrive_tick = last_port_arrive_tick
        for loc_idx in range(last_loc_idx + 1, last_loc_idx + stop_num + 1):
            port_idx, distance = route_points[loc_idx % route_length]
            arrive_tick += duration + ceil(distance / speed)
            predicted_future_stops.append(
                Stop(
                    port_idx=port_idx,
                    arrive_tick=arrive_tick,
                    leave_tick=arrive_tick + duration
                )
            )

        return predicted_future_stops

    def _parse(self, route_info: RoutesInfo, port_mapping: dict):
        idx = 0
        route_mapping = route_info.mapping

        for vessel_name, vessel_node in self._vessel_conf.items():
            self.mapping[vessel_name] = idx

            #
            route_name = vessel_node["route"]["route_name"]
            start_port_idx = port_mapping[vessel_node["route"]["initial_port_name"]]
            route_points = route_info.get_route_points_by_name(route_name)
            route_length = len(route_points)
            loc_idx_in_route = 0

            # find the start point
            while route_points[loc_idx_in_route][0] != start_port_idx:
                loc_idx_in_route += 1

            # update initial value fields
            self._initial_values["route"].append(route_mapping[route_name])
            self._initial_values["capacity"].append(vessel_node["capacity"])

            speed = vessel_node["sailing"]["speed"]
            speed_noise = vessel_node["sailing"]["noise"]
            duration = vessel_node["parking"]["duration"]
            duration_noise = vessel_node["parking"]["noise"]

            tick = 0
            period_no_noise = 0
            extra_stop_counter = 0

            # unpack the route by max tick and future stop number
            while extra_stop_counter <= self._stop_numbers[1]:
                port_id = route_points[loc_idx_in_route][0]
                parking_duration = ceil(apply_noise(duration, duration_noise, route_init_rand))

                if parking_duration <= 0:
                    raise VesselDurationWrong(
                        f"Invalid vessel parking duration time {parking_duration}, it must large than 0, please check "
                        f"your duration and noise.")

                stop = Stop(tick, tick + parking_duration, port_id)

                if len(self.unpacked_routes) == idx:
                    self.unpacked_routes.append([])

                self.unpacked_routes[idx].append(stop)

                # use distance and speed (all with noise) to calculate tick of arrival and departure
                distance = route_points[loc_idx_in_route][1]

                noised_speed = apply_noise(speed, speed_noise, route_init_rand)
                sailing_duration = ceil(distance / noised_speed)
                tick += parking_duration + sailing_duration
                whole_duration_no_noise = duration + ceil(distance/speed)
                period_no_noise += (whole_duration_no_noise if len(self.unpacked_routes[idx]) <= route_length else 0)
                loc_idx_in_route = (loc_idx_in_route + 1) % route_length
                extra_stop_counter += (1 if tick > self._max_tick else 0)

            self._initial_values["period"].append(period_no_noise)

            idx += 1
            self._length += 1


class EcrDataGenerator:
    """
    Data generator for ECR problem
    """
    def __init__(self, max_tick: int, config_path: str):
        self._max_tick = max_tick
        self._config = None
        self._pure_config = None

        with open(config_path, "r") as fp:
            self._config = safe_load(fp)

        self._ports: PortsInfo = None
        self._vessels: VesselsInfo = None
        self._routes: RoutesInfo = None
        self._node_mapping = {"static": {}, "dynamic": {}}

        self._parse_misc(self._config)
        self._parse_ports(self._config["ports"])
        self._parse_routes(self._config["routes"])
        self._parse_vessels(self._config["vessels"])
        self._parse_container_proportion(self._config["container_usage_proportion"])

        # fill the node mapping
        for port_name, port_idx in self._ports.mapping.items():
            self._node_mapping["static"][port_idx] = port_name

        for vessel_name, vessel_idx in self._vessels.mapping.items():
            self._node_mapping["dynamic"][vessel_idx] = vessel_name

    @property
    def container_volume(self) -> float:
        """
        Volume of each container, we only support one size now
        """
        return self._container_volume

    @property
    def port_num(self) -> int:
        """
        Number of port configured
        """
        return len(self._ports)

    @property
    def vessel_num(self) -> float:
        """
        Number of vessel configured
        """
        return len(self._vessels)

    @property
    def node_mapping(self) -> dict:
        """
        Name mapping for each node

        Returns:
            dictionary with "static" and "dynamic" as keys, value is name->index mapping
        """
        return self._node_mapping

    @property
    def vessel_stops(self) -> list:
        """
        Stops of all the vessels

        Returns:
            a list of stop list, that the index is vessel index
        """
        return self._vessels.unpacked_routes

    @property
    def stop_number(self) -> tuple:
        """
        A tuple (past number, future number) of stop numbers
        """
        return self._stop_numbers

    def generate_orders(self, tick: int, total_empty_container: int):
        """
        Generate orders by specified ticks

        Args:
            tick (int): which tick the new orders belongs to
            total_empty_container (int): how many available empty container we have now

        Returns:
            a list of Order object
        """
        order_list = []
        orders_to_gen = int(self._container_proportion.order_distribution[tick])

        if self._order_generate_mode == OrderGenerateMode.UNFIXED:
            delta = self._total_container_number - total_empty_container

            if orders_to_gen <= delta:
                return order_list

            orders_to_gen -= self._total_container_number - total_empty_container

        remaining_orders = orders_to_gen  # used to make sure all the order generated

        # collect and apply noise on the source distribution, then normalized it, to make sure our total number is same
        noised_source_dist = []

        for port_idx in range(self.port_num):
            source_dist, _ = self._ports.get_order_distribution(port_idx)

            noised_source_dist.append(apply_noise(source_dist[0], source_dist[1], order_num_rand))

        noised_source_dist = sum_normalize(noised_source_dist)

        for port_idx in range(self.port_num):
            if remaining_orders == 0:
                break

            _, targets_dist = self._ports.get_order_distribution(port_idx)

            # apply noise and normalize
            noised_targets_dist = sum_normalize([apply_noise(target[1], target[2], order_num_rand) for target in targets_dist])

            cur_port_order_num = ceil(orders_to_gen * noised_source_dist[port_idx])

            # make sure the total number is correct
            cur_port_order_num = min(cur_port_order_num, remaining_orders)
            remaining_orders -= cur_port_order_num

            if cur_port_order_num > 0:
                target_remaining_orders = cur_port_order_num

                for i, target in enumerate(targets_dist):
                    cur_num = ceil(cur_port_order_num * noised_targets_dist[i])

                    cur_num = min(cur_num, target_remaining_orders)
                    target_remaining_orders -= cur_num

                    if cur_num > 0:
                        order = Order(tick, port_idx, target[0], cur_num)

                        order_list.append(order)

        # TODO: remove later
        assert sum([o.quantity for o in order_list]) == orders_to_gen

        return order_list

    @property
    def port_initial_info(self):
        """
        Initial value of ports

        A dictionary with "empty" and "capacity" as key, value is list of value (index is the port idx)
        """
        return self._ports.initial_value

    @property
    def vessel_initial_info(self):
        """
        Initial value of vessels.

        A dictionary with "route" and "capacity" as key, value is list of value (index is vessel index)
        """
        return self._vessels.initial_values

    def _clean_config(self, config_dict: dict, remove_list: list):
        if type(config_dict) == dict:
            produced_config_dict = {}
            for key, value in config_dict.items():
                safe = True
                for word in remove_list:
                    if word in key:
                        safe = False
                        break
                if safe:
                    produced_config_dict[key] = self._clean_config(value, remove_list)
            return produced_config_dict
        else:
            return config_dict

    def get_pure_config(self):
        if self._pure_config is None:
            self._pure_config = self._clean_config(self._config, ['noise', 'order', 'proportion'])
        return self._pure_config

    def get_stop_from_idx(self, vessel_idx: int, loc_idx: int):
        """
        Get stop details for a vessel

        Args:
            vessel_idx (int): index of vessel
            loc_idx (int): index of location

        Returns:
            A Stop object
        """
        return self._vessels.get_stop(vessel_idx, loc_idx)

    def get_stop_list(self, vessel_idx: int, last_loc_idx: int, next_loc_idx: int):
        """
        Get a past and future stop list

        Args:
            vessel_idx (int): index of vessel
            last_loc_idx (int): index of last location
            next_loc_idx (int): index of next location

        Returns:
            tuple of ([past stops], [future stops])
        """
        return self._vessels.get_stops_list(vessel_idx, last_loc_idx, next_loc_idx)

    def get_full_buffer_tick(self, port_idx: int):
        """
        Get the buffer time for full return (with noise)

        Args:
            port_idx (int): port index to get

        Returns:
            buffer time (int)
        """
        return self._ports.get_full_return_buffer_tick(port_idx)

    def get_empty_buffer_tick(self, port_idx: int):
        """
        Get the buffer time for empty return (with noise)

        Args:
            port_idx (int): port index to get

        Returns:
            buffer time (int)
        """
        return self._ports.get_empty_return_buffer_tick(port_idx)

    def get_reachable_stops(self, vessel_idx: int, route_idx: int, next_loc_idx: int):
        """
        Get a list of stops that the vessel can arrive (as order destination)

        Args:
            vessel_idx (int): index of vessel
            route_idx (int): index of route
            next_loc_idx (int): index of next location

        Returns:
            A list of Stop object
        """
        route_length = self._routes.length_dict[route_idx]
        stops = self._vessels.unpacked_routes[vessel_idx][next_loc_idx + 1: next_loc_idx + 1 + route_length]

        return [(stop.port_idx, stop.arrive_tick) for stop in stops]

    def get_planed_stops(self, vessel_idx: int, route_idx: int, next_loc_idx: int):
        """
        Get following planed stop (without noise) for vessel, used for predict

        Returns:
            list of Tuple (port_idx, arrive_tick)
        """
        route_length = self._routes.length_dict[route_idx]

        stops = self._vessels.predict_future_stops_from_loc(vessel_idx, next_loc_idx, route_length)

        return [(stop.port_idx, stop.arrive_tick) for stop in stops]

    def get_vessel_period(self, vessel_idx: int):
        """
        Get route period (in ticks) of specified vessel

        Returns:
            Period of specified vessel
        """
        return self._vessels.initial_values["period"][vessel_idx]

    def _parse_misc(self, config):
        self._total_container_number = config["total_containers"]
        self._container_volume = config["container_volumes"][0]
        self._order_generate_mode = OrderGenerateMode(config["order_generate_mode"])
        self._stop_numbers = config["stop_number"]

    def _parse_container_proportion(self, proportion_conf):
        self._container_proportion = GlobalOrderProportion(proportion_conf, self._max_tick,
                                                           self._total_container_number)

    def _parse_ports(self, ports_conf):
        self._ports = PortsInfo(ports_conf, self._total_container_number)

    def _parse_vessels(self, vessels_conf):
        self._vessels = VesselsInfo(self._stop_numbers, vessels_conf, self._ports.mapping, self._routes, self._max_tick)

    def _parse_routes(self, route_conf):
        self._routes = RoutesInfo(route_conf, self._ports.mapping)
