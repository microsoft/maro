# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import csv
import os
from typing import List

import numpy as np
from yaml import safe_dump

from .cim_data_generator import CimDataGenerator
from .entities import CimSyntheticDataCollection


class CimDataDumpUtil:
    """Utilities to dump cim data from data collection, it will generate following files in specified folder:
    ports.csv, vessels.csv, stops.csv, routes.csv, order_proportion.csv, global_order_proportion.txt, misc.yml

    Args:
        data_collection (CimSyntheticDataCollection): Data collection to dump.
    """

    def __init__(self, data_collection: CimSyntheticDataCollection):
        self._data_collection = data_collection

    def dump(self, output_folder: str):
        """Dump cim data into specified folder.

        Args:
            output_folder (str): Folder to save dumped files.

        """
        # mapping for quick accessing
        vessel_idx2name_dict = {idx: name for name, idx in self._data_collection.vessel_mapping.items()}
        port_idx2name_dict = {idx: name for name, idx in self._data_collection.port_mapping.items()}
        route_idx2name_dict = {idx: name for name, idx in self._data_collection.route_mapping.items()}

        # dump files
        self._dump_stops(output_folder, vessel_idx2name_dict, port_idx2name_dict)
        self._dump_ports(output_folder)
        self._dump_vessels(output_folder)
        self._dump_routes(output_folder, route_idx2name_dict)
        self._dump_order_proportions(output_folder, port_idx2name_dict)
        self._dump_misc(output_folder)
        self._dump_global_order_proportions(output_folder)

    def _dump_global_order_proportions(self, output_folder: str):
        """global_order_proportion.txt"""
        global_order_prop_file = os.path.join(output_folder, "global_order_proportion.txt")

        np.savetxt(global_order_prop_file, self._data_collection.order_proportion)

    def _dump_stops(self, output_folder: str, vessel_idx2name_dict: dict, port_idx2name_dict: dict):
        """
        stops.csv: (stops.metal.yml)
            vessel_name, vessel_index, port_name, port_index, arrival_tick, departure_tick
        """
        stops_file_path = os.path.join(output_folder, "stops.csv")
        headers = ["vessel_name", "vessel_index", "port_name", "port_index", "arrival_tick", "departure_tick"]

        def stop_generator():
            for vessel_stops in self._data_collection.vessel_stops:
                for stop in vessel_stops:
                    yield [
                        vessel_idx2name_dict[stop.vessel_idx],
                        stop.vessel_idx,
                        port_idx2name_dict[stop.port_idx],
                        stop.port_idx,
                        stop.arrival_tick,
                        stop.leave_tick
                    ]

        self._dump_csv_file(stops_file_path, headers, stop_generator)

    def _dump_ports(self, output_folder: str):
        """
        ports.csv:
            index, name, capacity, empty, source_order_proportion, empty_return_buffer, full_return_buffer
        """

        ports_file_path = os.path.join(output_folder, "ports.csv")
        headers = [
            "index", "name", "capacity", "empty", "order_proportion", "order_proportion_noise",
            "empty_return_buffer", "empty_return_buffer_noise", "full_return_buffer", "full_return_buffer_noise"
        ]

        def port_generator():
            for port in self._data_collection.port_settings:
                yield [
                    port.index,
                    port.name,
                    port.capacity,
                    port.empty,
                    port.source_proportion.base,
                    port.source_proportion.noise,
                    port.empty_return_buffer.base,
                    port.empty_return_buffer.noise,
                    port.full_return_buffer.base,
                    port.full_return_buffer.noise
                ]

        self._dump_csv_file(ports_file_path, headers, port_generator)

    def _dump_vessels(self, output_folder: str):
        """
        vessels.csv
            index, name, capacity, route_name, route_index, start_port_name,
            start_port_index, sailing_speed, sailing_speed_noise, parking_duration, parking_noise

        """
        vessels_file_path = os.path.join(output_folder, "vessels.csv")
        headers = [
            "index", "name", "capacity", "route_name", "route_index", "start_port_name", "start_port_index",
            "sailing_speed", "sailing_speed_noise", "parking_duration", "parking_noise", "period", "empty"
        ]

        route_mapping = self._data_collection.route_mapping
        port_mapping = self._data_collection.port_mapping
        vessels = self._data_collection.vessel_settings
        vessel_period = self._data_collection.vessel_period_without_noise

        def vessel_generator():
            for vessel in vessels:
                yield [
                    vessel.index,
                    vessel.name,
                    vessel.capacity,
                    vessel.route_name,
                    route_mapping[vessel.route_name],
                    vessel.start_port_name,
                    port_mapping[vessel.start_port_name],
                    vessel.sailing_speed,
                    vessel.sailing_noise,
                    vessel.parking_duration,
                    vessel.parking_noise,
                    vessel_period[vessel.index],
                    vessel.empty
                ]

        self._dump_csv_file(vessels_file_path, headers, vessel_generator)

    def _dump_routes(self, output_folder: str, route_idx2name_dict: dict):
        """
        routes.csv -> used to get vessel plan (without noise)
            index, name, port_name, port_index, distance_to_next_port
        """
        routes_file_path = os.path.join(output_folder, "routes.csv")
        headers = ["index", "name", "port_name", "port_index", "distance_to_next_port"]

        routes = self._data_collection.routes
        port_mapping = self._data_collection.port_mapping

        def route_generator():
            for route in routes:
                for point in route:
                    yield [
                        point.index,
                        route_idx2name_dict[point.index],
                        point.port_name,
                        port_mapping[point.port_name],
                        point.distance_to_next_port
                    ]

        self._dump_csv_file(routes_file_path, headers, route_generator)

    def _dump_order_proportions(self, output_folder: str, port_idx2name_dict: dict):
        """
        target_order_proportions.csv
            source_port_index, target_port_index, order_proportion, proportion_noise
        """

        proportion_file_path = os.path.join(output_folder, "order_proportion.csv")
        headers = [
            "source_port_name", "source_port_index", "dest_port_name",
            "dest_port_index", "proportion", "proportion_noise"
        ]

        ports = self._data_collection.port_settings

        def order_prop_generator():
            for port in ports:
                for prop in port.target_proportions:
                    yield [
                        port.name,
                        port.index,
                        port_idx2name_dict[prop.index],
                        prop.index,
                        prop.base,
                        prop.noise
                    ]

        self._dump_csv_file(proportion_file_path, headers, order_prop_generator)

    def _dump_misc(self, output_folder: str):
        """
        order mode, total container, container volume, and other misc items with yaml format
        """
        misc_items = {
            "order_mode": self._data_collection.order_mode.value,
            "total_container": self._data_collection.total_containers,
            "past_stop_number": self._data_collection.past_stop_number,
            "future_stop_number": self._data_collection.future_stop_number,
            "container_volume": self._data_collection.container_volume,
            "load_cost_factor": self._data_collection.load_cost_factor,
            "dsch_cost_factor": self._data_collection.dsch_cost_factor,
            "max_tick": self._data_collection.max_tick,
            "seed": self._data_collection.seed,
            "version": self._data_collection.version
        }

        misc_file_path = os.path.join(output_folder, "misc.yml")

        with open(misc_file_path, "wt+") as fp:
            safe_dump(misc_items, fp)

    def _dump_csv_file(self, file_path: str, headers: List[str], line_generator: callable):
        """helper method to dump csv file

        Args:
            file_path(str): path of output csv file
            headers(List[str]): list of header
            line_generator(callable): generator function to generate line to write
        """
        with open(file_path, "wt+", newline="") as fp:
            writer = csv.writer(fp)

            writer.writerow(headers)

            for line in line_generator():
                writer.writerow(line)


def dump_from_config(config_file: str, output_folder: str, max_tick: int):
    """Dump cim data from config, this will call data generator to generate data , and dump it.

    NOTE:
        This function will not convert csv files into binary.

    Args:
        config_file (str): Configuration path.
        output_folder (str): Output folder to save files.
        max_tick(int): Max tick to gen.
    """
    assert config_file is not None and os.path.exists(config_file), f"Got config file path: {config_file}"
    assert output_folder is not None and os.path.exists(output_folder), f"Got output folder path: {output_folder}"
    assert max_tick is not None and max_tick > 0, f"Got max tick: {max_tick}"

    generator = CimDataGenerator()

    data_collection = generator.gen_data(config_file, max_tick=max_tick, start_tick=0)

    dump_util = CimDataDumpUtil(data_collection)

    dump_util.dump(output_folder)
