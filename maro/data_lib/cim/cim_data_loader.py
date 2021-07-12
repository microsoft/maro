# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import csv
import math
import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List

import numpy as np
from yaml import safe_load

from maro.data_lib import BinaryReader

from .entities import (
    CimBaseDataCollection, CimRealDataCollection, CimSyntheticDataCollection, NoisedItem, Order, OrderGenerateMode,
    PortSetting, RoutePoint, Stop, SyntheticPortSetting, VesselSetting
)


class CimBaseDataLoader(ABC):
    """Utility to load data from dump folder and real input data folder."""

    @abstractmethod
    def load(self, data_folder: str) -> CimBaseDataCollection:
        pass

    def _load_misc(self, data_folder: str) -> dict:
        """Load misc items from yaml"""
        misc_file_path = os.path.join(data_folder, "misc.yml")
        for _ in range(3):
            if not os.path.exists(misc_file_path):
                time.sleep(10)
        with open(misc_file_path, "rt") as fp:
            return safe_load(fp)

    def _read_csv_lines(self, file_path: str):
        """Helper to read and yield line from csv file"""
        for _ in range(3):
            if not os.path.exists(file_path):
                time.sleep(10)
        with open(file_path, "rt") as fp:
            reader = csv.DictReader(fp)

            for line in reader:
                yield line

    def _load_vessels(self, data_folder: str) -> (Dict[str, int], List[VesselSetting]):
        vessel_mapping: Dict[str, int] = {}
        vessels: List[VesselSetting] = []

        vessels_file_path = os.path.join(data_folder, "vessels.csv")

        for line in self._read_csv_lines(vessels_file_path):
            vessel_name = line["name"]
            vessel_index = int(line["index"])

            vessel_mapping[vessel_name] = vessel_index

            vessel = VesselSetting(
                vessel_index,
                vessel_name,
                int(line["capacity"]),
                line["route_name"],
                line["start_port_name"],
                float(line["sailing_speed"]),
                float(line["sailing_speed_noise"]),
                int(line["parking_duration"]),
                float(line["parking_noise"]),
                int(line["empty"])
            )

            vessels.append(vessel)

        return vessel_mapping, vessels

    def _load_vessel_period(self, data_folder: str) -> List[int]:
        vessels_file_path = os.path.join(data_folder, "vessels.csv")

        periods_without_noise: List[int] = []
        for line in self._read_csv_lines(vessels_file_path):
            periods_without_noise.append(int(line["period"]))

        return periods_without_noise

    def _calculate_vessel_period(
        self, vessels: List[VesselSetting], routes: List[List[RoutePoint]], route_mapping: Dict[str, int]
    ) -> List[int]:
        expected_periods: List[int] = []
        for vessel in vessels:
            route_points = routes[route_mapping[vessel.route_name]]

            expected_period = 0
            for route_point in route_points:
                expected_period += (
                    vessel.parking_duration
                    + math.ceil(route_point.distance_to_next_port / vessel.sailing_speed)
                )
            expected_periods.append(expected_period)

        return expected_periods

    def _load_routes(self, data_folder: str) -> (Dict[str, int], List[List[RoutePoint]]):
        route_mapping: Dict[str, int] = {}
        routes: List[List[RoutePoint]] = []

        route_file_path = os.path.join(data_folder, "routes.csv")

        for line in self._read_csv_lines(route_file_path):
            route_index = int(line["index"])
            route_name = line["name"]

            route_mapping[route_name] = route_index

            if route_index >= len(routes):
                routes.append([])

            route_point = RoutePoint(
                route_index, line["port_name"], float(line["distance_to_next_port"]))

            routes[route_index].append(route_point)

        return route_mapping, routes

    def _load_stops(self, data_folder: str, vessel_number: int) -> List[List[Stop]]:
        bin_path = os.path.join(data_folder, "stops.bin")
        if os.path.exists(bin_path):
            return self._load_stops_from_bin(bin_path, vessel_number)
        else:
            print(f"No stops binary file in {data_folder}, read from csv file instead...")
            csv_path = os.path.join(data_folder, "stops.csv")
            return self._load_stops_from_csv(csv_path, vessel_number)

    def _load_stops_from_csv(self, stops_file_path: str, vessel_number: int) -> List[List[Stop]]:
        stops: List[List[Stop]] = []

        for _ in range(vessel_number):
            stops.append([])

        for line in self._read_csv_lines(stops_file_path):
            vessel_stops: List[Stop] = stops[int(line["vessel_index"])]

            stop = Stop(
                len(vessel_stops),
                int(line["arrival_tick"]),
                int(line["departure_tick"]),
                int(line["port_index"]),
                int(line["vessel_index"])
            )

            vessel_stops.append(stop)

        return stops

    def _load_stops_from_bin(self, stops_file_path: str, vessel_number: int) -> List[List[Stop]]:
        stops: List[List[Stop]] = []

        for _ in range(vessel_number):
            stops.append([])

        reader = BinaryReader(stops_file_path)

        for stop_item in reader.items():
            vessel_stops: List[Stop] = stops[stop_item.vessel_index]

            stop = Stop(len(vessel_stops),
                        stop_item.timestamp,
                        stop_item.leave_tick,
                        stop_item.port_index,
                        stop_item.vessel_index)

            vessel_stops.append(stop)

        return stops


class CimDumpDataLoader(CimBaseDataLoader):
    """Utility to load data from dump folder"""

    def load(self, data_folder: str) -> CimSyntheticDataCollection:
        """Load data from dump folder

        NOTE:
        dumps folder should contains following files.
        ports.csv, vessels.csv, routes.csv, order_proportion.csv,
        global_order_proportion.txt, misc.yml, stops.bin

        Args:
            data_folders(str): folder that contains dumped files

        Returns:
            CimSyntheticDataCollection: data collection for data container
        """
        # load from files
        misc_items = self._load_misc(data_folder)
        order_target_proportion = self._load_order_proportions(data_folder)
        port_mapping, ports = self._load_ports(data_folder, order_target_proportion)
        route_mapping, routes = self._load_routes(data_folder)
        vessel_mapping, vessels = self._load_vessels(data_folder)
        periods_without_noise = self._load_vessel_period(data_folder)
        stops = self._load_stops(data_folder, len(vessels))
        global_order_proportions = self._load_global_order_proportions(data_folder)

        # construct data collection
        # NOTE: this is a namedtuple, so out-side cannot change it
        data_collection = CimSyntheticDataCollection(
            port_settings=ports,
            port_mapping=port_mapping,
            vessel_settings=vessels,
            vessel_mapping=vessel_mapping,
            vessel_stops=stops,
            routes=routes,
            route_mapping=route_mapping,
            vessel_period_without_noise=periods_without_noise,
            container_volume=misc_items["container_volume"],
            load_cost_factor=misc_items["load_cost_factor"],
            dsch_cost_factor=misc_items["dsch_cost_factor"],
            past_stop_number=misc_items["past_stop_number"],
            future_stop_number=misc_items["future_stop_number"],
            max_tick=misc_items["max_tick"],
            seed=misc_items["seed"],
            total_containers=misc_items["total_container"],
            order_mode=OrderGenerateMode(misc_items["order_mode"]),
            order_proportion=global_order_proportions,
            version=misc_items["version"]
        )

        return data_collection

    def _load_global_order_proportions(self, data_folder: str) -> np.ndarray:
        """load global order proportions from txt file"""
        global_order_prop_file = os.path.join(
            data_folder, "global_order_proportion.txt")

        return np.loadtxt(global_order_prop_file)

    def _load_order_proportions(self, data_folder: str) -> Dict[int, List[NoisedItem]]:
        """Load target order proportions from file"""
        target_proportions: Dict[int, List[NoisedItem]] = defaultdict(list)

        proportion_file_path = os.path.join(data_folder, "order_proportion.csv")

        for line in self._read_csv_lines(proportion_file_path):
            source_port_index = int(line["source_port_index"])

            target_prop = NoisedItem(
                int(line["dest_port_index"]),
                float(line["proportion"]),
                float(line["proportion_noise"])
            )

            target_proportions[source_port_index].append(target_prop)

        return target_proportions

    def _load_ports(self, data_folder: str, order_target_proportion: dict) -> dict:
        ports_file_path = os.path.join(data_folder, "ports.csv")

        port_mapping: Dict[str, int] = {}
        ports: List[SyntheticPortSetting] = []

        for line in self._read_csv_lines(ports_file_path):
            port_name = line["name"]
            port_index = int(line["index"])

            port_mapping[port_name] = port_index

            full_rtn_buffer = NoisedItem(
                port_index,
                int(line["full_return_buffer"]),
                int(line["full_return_buffer_noise"]))

            empty_rtn_buffer = NoisedItem(
                port_index,
                int(line["empty_return_buffer"]),
                int(line["empty_return_buffer_noise"]))

            source_order_proportion = NoisedItem(
                port_index,
                float(line["order_proportion"]),
                float(line["order_proportion_noise"])
            )

            port = SyntheticPortSetting(
                port_index,
                port_name,
                int(line["capacity"]),
                int(line["empty"]),
                empty_rtn_buffer,
                full_rtn_buffer,
                source_order_proportion,
                order_target_proportion[port_index]
            )

            ports.append(port)

        return port_mapping, ports


class CimRealDataLoader(CimBaseDataLoader):
    """Utility to load data from data folder"""

    def load(self, data_folder: str) -> CimRealDataCollection:
        """Load data from data folder

        NOTE:
        data folder should contains following files.
        ports.csv, vessels.csv, routes.csv, order.csv,
        misc.yml, stops.csv

        Args:
            data_folders(str): folder that contains data files.

        Returns:
            CimDataCollection: data collection for data container
        """
        # load from files
        misc_items = self._load_misc(data_folder)
        port_mapping, ports = self._load_ports(data_folder)
        route_mapping, routes = self._load_routes(data_folder)
        vessel_mapping, vessels = self._load_vessels(data_folder)
        periods_without_noise = self._calculate_vessel_period(vessels, routes, route_mapping)
        stops = self._load_stops(data_folder, len(vessels))
        orders = self._load_orders(data_folder)

        # construct data collection
        # NOTE: this is a namedtuple, so out-side cannot change it
        data_collection = CimRealDataCollection(
            port_settings=ports,
            port_mapping=port_mapping,
            vessel_settings=vessels,
            vessel_mapping=vessel_mapping,
            vessel_stops=stops,
            routes=routes,
            route_mapping=route_mapping,
            vessel_period_without_noise=periods_without_noise,
            container_volume=misc_items["container_volume"],
            load_cost_factor=misc_items["load_cost_factor"],
            dsch_cost_factor=misc_items["dsch_cost_factor"],
            past_stop_number=misc_items["past_stop_number"],
            future_stop_number=misc_items["future_stop_number"],
            max_tick=misc_items["max_tick"],
            seed=misc_items["seed"],
            orders=orders
        )

        return data_collection

    def _load_ports(self, data_folder: str) -> dict:
        ports_file_path = os.path.join(data_folder, "ports.csv")

        port_mapping: Dict[str, int] = {}
        ports: List[PortSetting] = []

        for line in self._read_csv_lines(ports_file_path):
            port_index = int(line["index"])
            port_name = line["name"]

            port_mapping[port_name] = port_index

            full_rtn_buffer = NoisedItem(
                port_index,
                int(line["full_return_buffer"]),
                int(line["full_return_buffer_noise"])
            )

            empty_rtn_buffer = NoisedItem(
                port_index,
                int(line["empty_return_buffer"]),
                int(line["empty_return_buffer_noise"])
            )

            port = PortSetting(
                port_index,
                port_name,
                int(line["capacity"]),
                int(line["empty"]),
                empty_rtn_buffer,
                full_rtn_buffer
            )

            ports.append(port)

        return port_mapping, ports

    def _load_orders(self, data_folder: str) -> Dict[int, List[Order]]:
        bin_path = os.path.join(data_folder, "orders.bin")
        if os.path.exists(bin_path):
            return self._load_orders_from_bin(bin_path)
        else:
            print(f"No orders binary file in {data_folder}, read from csv file instead...")
            csv_path = os.path.join(data_folder, "orders.csv")
            return self._load_orders_from_csv(csv_path)

    def _load_orders_from_csv(self, order_file_path: str) -> Dict[int, List[Order]]:
        orders: Dict[int, List[Order]] = {}

        for line in self._read_csv_lines(order_file_path):
            tick = int(line["tick"])
            if tick not in orders:
                orders[tick] = []
            orders[tick].append(
                Order(
                    int(line["tick"]),
                    int(line["source_port_index"]),
                    int(line["dest_port_index"]),
                    int(line["quantity"])
                )
            )

        return orders

    def _load_orders_from_bin(self, order_file_path: str) -> Dict[int, List[Order]]:
        orders: Dict[int, List[Order]] = {}

        reader = BinaryReader(order_file_path)

        for order in reader.items():
            tick = order.timestamp
            if tick not in orders:
                orders[tick] = []
            orders[tick].append(
                Order(
                    order.timestamp,
                    order.src_port_index,
                    order.dest_port_index,
                    order.quantity
                )
            )

        return orders


def load_from_folder(source_folder: str) -> CimSyntheticDataCollection:
    """Load data from dump folder.

    NOTE:
        Dumps folder should contains following files:
    ports.csv, vessels.csv, routes.csv, order_proportion.csv,
    global_order_proportion.txt, misc.yml, stops.bin.

    Args:
        source_folder(str): Source folder container dumped files.

    Returns:
        CimSyntheticDataCollection: Data collection for cim data container.
    """
    loader = CimDumpDataLoader()

    return loader.load(source_folder)


def load_real_data_from_folder(source_folder: str) -> CimRealDataCollection:
    """Load real data from folder.

    NOTE:
        Real data folder should contains following files:
    ports.csv, vessels.csv, routes.csv, misc.yml, stops.csv, orders.csv.

    Args:
        source_folder(str): Source folder contains data files.

    Returns:
        CimRealDataCollection: Data collection for cim data container.
    """
    loader = CimRealDataLoader()

    return loader.load(source_folder)
