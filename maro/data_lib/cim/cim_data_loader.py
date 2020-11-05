# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import csv
import os
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np
from yaml import safe_load

from maro.data_lib import BinaryReader

from .entities import CimDataCollection, NoisedItem, OrderGenerateMode, PortSetting, RoutePoint, Stop, VesselSetting


class CimDumpDataLoader:
    """Utility to load data from dump folder"""

    def load(self, dumps_folder: str) -> CimDataCollection:
        """Load data from dump folder

        NOTE:
        dumps folder should contains following files.
        ports.csv, vessels.csv, routes.csv, order_proportion.csv,
        global_order_proportion.txt, misc.yml, stops.bin

        Args:
            dumps_folders(str): folder that contains dumped files

        Returns:
            CimDataCollection: data collection for data container
        """
        # load from files
        misc_items = self._load_misc(dumps_folder)
        order_target_proportion = self._load_order_proportions(dumps_folder)
        port_mapping, ports = self._load_ports(dumps_folder, order_target_proportion)
        route_mapping, routes = self._load_routes(dumps_folder)
        vessel_mapping, vessels, peroids_without_noise = self._load_vessels(dumps_folder)
        stops = self._load_stops(dumps_folder, len(vessels))
        global_order_proportions = self._load_global_order_proportions(dumps_folder)

        # construct data collection
        # NOTE: this is a namedtuple, so out-side cannot change it
        data_collection = CimDataCollection(
            misc_items["total_container"],
            misc_items["past_stop_number"],
            misc_items["future_stop_number"],
            misc_items["container_volume"],
            OrderGenerateMode(misc_items["order_mode"]),
            ports,
            port_mapping,
            vessels,
            vessel_mapping,
            stops,
            global_order_proportions,
            routes,
            route_mapping,
            peroids_without_noise,
            misc_items["max_tick"],
            misc_items["seed"],
            misc_items["version"]
        )

        return data_collection

    def _load_misc(self, dumps_folder: str) -> dict:
        """Load misc items from yaml"""
        misc_file_path = os.path.join(dumps_folder, "misc.yml")
        for _ in range(3):
            if not os.path.exists(misc_file_path):
                time.sleep(10)
        with open(misc_file_path, "rt") as fp:
            return safe_load(fp)

    def _load_global_order_proportions(self, dumps_folder: str) -> np.ndarray:
        """load global order proportions from txt file"""
        global_order_prop_file = os.path.join(
            dumps_folder, "global_order_proportion.txt")

        return np.loadtxt(global_order_prop_file)

    def _read_csv_lines(self, file_path: str):
        """Helper to read and yield line from csv file"""
        for _ in range(3):
            if not os.path.exists(file_path):
                time.sleep(10)
        with open(file_path, "rt") as fp:
            reader = csv.DictReader(fp)

            for line in reader:
                yield line

    def _load_order_proportions(self, dumps_folder: str) -> Dict[int, List[NoisedItem]]:
        """Load target order proportions from file"""
        target_proportions: Dict[int, List[NoisedItem]] = defaultdict(list)

        proportion_file_path = os.path.join(dumps_folder, "order_proportion.csv")

        for line in self._read_csv_lines(proportion_file_path):
            source_port_index = int(line["source_port_index"])

            target_prop = NoisedItem(
                int(line["dest_port_index"]),
                float(line["proportion"]),
                float(line["proportion_noise"])
            )

            target_proportions[source_port_index].append(target_prop)

        return target_proportions

    def _load_ports(self, dumps_folder: str, order_target_proportion: dict) -> dict:
        ports_file_path = os.path.join(dumps_folder, "ports.csv")

        port_mapping: Dict[str, int] = {}
        ports: List[PortSetting] = []

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

            port = PortSetting(
                port_index,
                port_name,
                int(line["capacity"]),
                int(line["empty"]),
                source_order_proportion,
                order_target_proportion[port_index],
                empty_rtn_buffer,
                full_rtn_buffer
            )

            ports.append(port)

        return port_mapping, ports

    def _load_vessels(self, dumps_folder: str) -> (Dict[str, int], List[VesselSetting]):
        vessel_mapping: Dict[str, int] = {}
        vessels: List[VesselSetting] = []
        periods_without_noise: List[int] = []

        vessels_file_path = os.path.join(dumps_folder, "vessels.csv")

        for line in self._read_csv_lines(vessels_file_path):
            vessel_name = line["name"]
            vessel_index = int(line["index"])

            vessel_mapping[vessel_name] = vessel_index
            periods_without_noise.append(int(line["period"]))

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

        return vessel_mapping, vessels, periods_without_noise

    def _load_routes(self, dumps_folder: str) -> (Dict[str, int], List[List[RoutePoint]]):
        route_mapping: Dict[str, int] = {}
        routes: List[List[RoutePoint]] = []

        route_file_path = os.path.join(dumps_folder, "routes.csv")

        for line in self._read_csv_lines(route_file_path):
            route_index = int(line["index"])
            route_name = line["name"]

            route_mapping[route_name] = route_index

            if route_index >= len(routes):
                routes.append([])

            route_point = RoutePoint(
                route_index, line["port_name"], float(line["distance"]))

            routes[route_index].append(route_point)

        return route_mapping, routes

    def _load_stops(self, dumps_folder: str, vessel_number: int) -> List[List[Stop]]:
        stops: List[List[Stop]] = []

        for _ in range(vessel_number):
            stops.append([])

        stops_file_path = os.path.join(dumps_folder, "stops.bin")

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


def load_from_folder(source_folder: str) -> CimDataCollection:
    """Load data from dump folder.

    NOTE:
        Dumps folder should contains following files:
    ports.csv, vessels.csv, routes.csv, order_proportion.csv,
    global_order_proportion.txt, misc.yml, stops.bin.

    Args:
        source_folder(str): Source folder container dumped files.

    Returns:
        CimDataCollection: Data collection for cim data container.
    """
    loader = CimDumpDataLoader()

    return loader.load(source_folder)
