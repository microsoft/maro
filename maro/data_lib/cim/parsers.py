# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from math import floor
from typing import Dict, List, Union

import numpy as np

from maro.data_lib.cim.entities import NoisedItem, RoutePoint, SyntheticPortSetting, VesselSetting
from maro.data_lib.cim.utils import ORDER_INIT_RAND_KEY, apply_noise, clip
from maro.simulator.utils import random


def parse_vessels(conf: dict) -> (Dict[str, int], List[VesselSetting]):
    """Parse specified vessel configurations.

    Args:
        conf(dict): Configurations to parse.

    Returns:
        (Dict[str, int], List[VesselSetting]): Vessel mappings (name to index), and settings list for all vessels.

    """
    mapping: Dict[str, int] = {}
    vessels: List[VesselSetting] = []

    index = 0

    for vessel_name, vessel_node in conf.items():
        mapping[vessel_name] = index

        sailing = vessel_node["sailing"]
        parking = vessel_node["parking"]
        route = vessel_node["route"]

        vessels.append(
            VesselSetting(
                index,
                vessel_name,
                vessel_node["capacity"],
                route["route_name"],
                route["initial_port_name"],
                sailing["speed"],
                sailing["noise"],
                parking["duration"],
                parking["noise"],
                # default no empty
                vessel_node.get("empty", 0)
            )
        )

        index += 1

    return mapping, vessels


def parse_global_order_proportion(conf: dict, total_container: int, max_tick: int, start_tick: int = 0) -> np.ndarray:
    """Parse specified configuration, and generate order proportion.

    Args:
        conf (dict): Configuration to parse.
        total_container (int): Total containers in this environment.
        max_tick (int): Max tick to generate.
        start_tick (int): Start tick to generate.

    Returns:
        np.ndarray: 1-dim numpy array for specified range.
    """
    durations: int = max_tick - start_tick

    order_proportion = np.zeros(durations, dtype="i")

    # read configurations
    period: int = conf["period"]
    noise: Union[float, int] = conf["sample_noise"]
    sample_nodes: list = [(x, y) for x, y in conf["sample_nodes"]]

    # step 1: interpolate with configured sample nodes to generate proportion in period

    # check if there is 0 and max_tick - 1 node exist ,insert if not exist
    if sample_nodes[0][0] != 0:
        sample_nodes.insert(0, (0, 0))
    if sample_nodes[-1][0] != period - 1:
        sample_nodes.append((period - 1, 0))

    # our xp is period
    xp = [node[0] for node in sample_nodes]
    yp = [node[1] for node in sample_nodes]

    # distribution per period
    order_period_distribution = np.interp(list(range(period)), xp, yp)

    # step 2: extend to specified range
    for t in range(start_tick, max_tick):
        orders = order_period_distribution[t % period]  # ratio

        # apply noise if the distribution not zero
        if orders != 0:
            if noise != 0:
                orders = apply_noise(orders, noise, random[ORDER_INIT_RAND_KEY])

            # clip and gen order
            orders = floor(clip(0, 1, orders) * total_container)
        order_proportion[t - start_tick] = orders

    return order_proportion


def parse_routes(conf: dict) -> (Dict[str, int], List[List[RoutePoint]]):
    """Parse specified route configuration.

    Args:
        conf (dict): Configuration to parse.

    Returns:
        (Dict[str, int], List[List[RoutePoint]]): Route mapping (name to index),
            and list of route point list (index is route index).
    """
    routes: List[List[RoutePoint]] = []
    route_mapping: Dict[str, int] = {}  # name->idx

    idx = 0

    for name, node in conf.items():
        route_mapping[name] = idx

        routes.append([RoutePoint(idx, n["port_name"], n["distance_to_next_port"]) for n in node])

        idx += 1

    return route_mapping, routes


def parse_ports(conf: dict, total_container: int) -> (Dict[str, int], List[SyntheticPortSetting]):
    """Parse specified port configurations.

    Args:
        conf (dict): Configuration to parse.
        total_container (int): Total container in current environment, used to calculate initial empty.

    Returns:
        (Dict[str, int], List[SyntheticPortSetting]): Port mapping (name to index), list of port settings.
    """
    # sum of ratio cannot large than 1
    total_ratio = sum([p["initial_container_proportion"] for p in conf.values()])

    assert round(total_ratio, 7) == 1

    ports_mapping: Dict[str, int] = {}

    index = 0

    # step 1: create mapping
    for port_name, port_info in conf.items():
        ports_mapping[port_name] = index
        index += 1

    port_settings: List[SyntheticPortSetting] = []

    for port_idx, port in enumerate(conf.items()):
        port_name, port_info = port

        # step 2: update initial values
        empty_ratio = port_info["initial_container_proportion"]

        # full return buffer configurations
        full_return_conf = port_info["full_return"]
        empty_return_conf = port_info["empty_return"]

        dist_conf = port_info["order_distribution"]
        source_dist_conf = dist_conf["source"]

        targets_dist = []

        # orders distribution to destination
        if "targets" in dist_conf:
            for target_port_name, target_conf in dist_conf["targets"].items():
                dist = NoisedItem(
                    ports_mapping[target_port_name],
                    target_conf["proportion"],
                    target_conf["noise"])

                targets_dist.append(dist)

        port_setting = SyntheticPortSetting(
            port_idx,
            port_name,
            port_info["capacity"],
            int(empty_ratio * total_container),
            NoisedItem(
                port_idx,
                empty_return_conf["buffer_ticks"],
                empty_return_conf["noise"]
            ),
            NoisedItem(
                port_idx,
                full_return_conf["buffer_ticks"],
                full_return_conf["noise"]
            ),
            NoisedItem(
                port_idx,
                source_dist_conf["proportion"],
                source_dist_conf["noise"]
            ),
            targets_dist
        )

        port_settings.append(port_setting)

    return ports_mapping, port_settings
