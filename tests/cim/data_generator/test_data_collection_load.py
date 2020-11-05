# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import tempfile
import unittest
from typing import Dict, List

from maro.data_lib import BinaryConverter
from maro.data_lib.cim import dump_from_config, load_from_folder
from maro.data_lib.cim.cim_data_dump import CimDataDumpUtil
from maro.data_lib.cim.cim_data_generator import CimDataGenerator
from maro.data_lib.cim.entities import CimDataCollection, NoisedItem, PortSetting, RoutePoint, Stop, VesselSetting

MAX_TICK = 20

class TestDumpsLoad(unittest.TestCase):
    def test_load_correct(self):
        config_path = os.path.join("tests", "data", "cim", "data_generator", "dumps", "config.yml")
        stops_meta_path = os.path.join("tests", "data", "cim", "data_generator" ,"dumps", "cim.stops.meta.yml")

        output_folder = tempfile.mkdtemp()

        # here we need to use CimDataDumpUtil manually to compare the result
        dc: CimDataCollection =  CimDataGenerator().gen_data(config_path, 20)

        dumper = CimDataDumpUtil(dc)

        dumper.dump(output_folder)

        # convert stops.csv into binary
        bconverter = BinaryConverter(os.path.join(output_folder, "stops.bin"), stops_meta_path)

        bconverter.add_csv(os.path.join(output_folder, "stops.csv"))
        bconverter.flush()

        # then load it
        dc2 = load_from_folder(output_folder)

        # compare
        self._compare_ports(dc, dc2)
        self._compare_vessels(dc, dc2)
        self._compare_stops(dc, dc2)
        self._compare_routes(dc, dc2)
        self._compare_misc(dc, dc2)
        self._compare_order_proportion(dc, dc2)

    def _compare_order_proportion(self, dc1: CimDataCollection, dc2: CimDataCollection):
        self.assertListEqual(list(dc1.order_proportion), list(dc2.order_proportion))

    def _compare_misc(self, dc1: CimDataCollection, dc2: CimDataCollection):
        self.assertEqual(dc1.max_tick, dc2.max_tick)
        self.assertEqual(dc1.order_mode, dc2.order_mode)
        self.assertEqual(dc1.total_containers, dc2.total_containers)
        self.assertEqual(dc1.cntr_volume, dc2.cntr_volume)

    def _compare_routes(self, dc1: CimDataCollection, dc2: CimDataCollection):
        routes_1 = dc1.routes
        routes_2 = dc2.routes

        self.assertTrue(len(routes_1) == len(routes_2))

        for ridx in range(len(routes_2)):
            route1_points: List[RoutePoint] = routes_1[ridx]
            route2_points: List[RoutePoint] = routes_2[ridx]

            self.assertTrue(len(route1_points) == len(route2_points))

            for pidx in range(len(route1_points)):
                p1: RoutePoint = route1_points[pidx]
                p2: RoutePoint = route2_points[pidx]

                self.assertTrue(p1.port_name == p2.port_name)
                self.assertTrue(p1.distance, p2.distance)

    def _compare_ports(self, dc1: CimDataCollection, dc2: CimDataCollection):
        ports_1 = dc1.ports_settings
        ports_2 = dc2.ports_settings

        self.assertTrue(len(ports_1) == len(ports_2))

        for port_index in range(len(ports_1)):
            port1: PortSetting = ports_1[port_index]
            port2: PortSetting = ports_2[port_index]

            self.assertTrue(port1.index == port2.index, f"{port1.index}, {port2.index}")
            self.assertTrue(port1.name == port2.name, f"{port1.name}, {port2.name}")
            self.assertTrue(port1.capacity == port2.capacity, f"{port1.capacity}, {port2.capacity}")
            self.assertTrue(port1.empty == port2.empty, f"{port1.empty}, {port2.empty}")
            self.assertTrue(port1.empty_return_buffer.base == port2.empty_return_buffer.base)
            self.assertTrue(port1.empty_return_buffer.noise == port2.empty_return_buffer.noise)
            self.assertTrue(port1.full_return_buffer.base == port2.full_return_buffer.base)
            self.assertTrue(port1.full_return_buffer.noise == port2.full_return_buffer.noise)
            self.assertTrue(port1.source_proportion.base == port2.source_proportion.base)
            self.assertTrue(port1.source_proportion.noise == port2.source_proportion.noise)
            self.assertTrue(len(port1.target_proportions) == len(port2.target_proportions))

            for tindex in range(len(port1.target_proportions)):
                tprop1: NoisedItem = port1.target_proportions[tindex]
                tprop2 = port2.target_proportions[tindex]

                self.assertTrue(tprop1.base == tprop2.base)
                self.assertTrue(tprop1.noise == tprop2.noise)


    def _compare_vessels(self, dc1: CimDataCollection, dc2: CimDataCollection):
        vessels_1: List[VesselSetting] = dc1.vessels_settings
        vessels_2: List[VesselSetting] = dc2.vessels_settings

        self.assertTrue(len(vessels_1) == len(vessels_2))

        for i in range(len(vessels_1)):
            vessel1: VesselSetting = vessels_1[i]
            vessel2: VesselSetting = vessels_2[i]

            self.assertTrue(vessel1.index == vessel2.index)
            self.assertTrue(vessel1.name == vessel2.name)
            self.assertTrue(vessel1.capacity == vessel2.capacity)
            self.assertTrue(vessel1.route_name == vessel2.route_name)
            self.assertTrue(vessel1.start_port_name == vessel2.start_port_name)
            self.assertTrue(vessel1.sailing_speed == vessel2.sailing_speed)
            self.assertTrue(vessel1.sailing_noise == vessel2.sailing_noise)
            self.assertTrue(vessel1.parking_duration == vessel2.parking_duration)
            self.assertTrue(vessel1.parking_noise == vessel2.parking_noise)


    def _compare_stops(self, dc1: CimDataCollection, dc2: CimDataCollection):
        stops_1: List[List[Stop]] = dc1.vessels_stops
        stops_2: List[List[Stop]] = dc2.vessels_stops

        self.assertTrue(len(stops_1) == len(stops_2))

        for vessel_index in range(len(stops_1)):

            vessel_stops_1 = stops_1[vessel_index]
            vessel_stops_2 = stops_2[vessel_index]

            self.assertTrue(len(vessel_stops_1) == len(vessel_stops_2))

            for stop_index in range(len(vessel_stops_1)):
                stop1: Stop = vessel_stops_1[stop_index]
                stop2: Stop = vessel_stops_2[stop_index]

                self.assertTrue(stop1.index == stop2.index, f"{stop1.index}, {stop2.index}")
                self.assertTrue(stop1.leave_tick == stop2.leave_tick)
                self.assertTrue(stop1.arrive_tick == stop2.arrive_tick)
                self.assertTrue(stop1.port_idx == stop2.port_idx)
                self.assertTrue(stop1.vessel_idx == stop2.vessel_idx)

if __name__=="__main__":
    unittest.main()
