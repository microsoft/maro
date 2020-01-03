# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os
import csv
from collections import defaultdict
from maro.simulator.scenarios.ecr.vessel import Vessel
from maro.simulator.scenarios.ecr.port import Port
from maro.simulator.scenarios.ecr.common import Stop, Order

PORT_NUM = 3
VESSEL_NUM = 2
PAST_STOP_NUM = 2
FUTURE_STOP_NUM = 3
TOTAL_CONTAINER_NUM = 100
CONTAINER_VOLUME = 1


class MockDataGenerator:
    '''
    this is a mock data generator to test business engine, 
    here we hard coded:
    1. port number
    2. vessel number
    '''

    def __init__(self, case_folder: str):

        assert os.path.exists(case_folder)

        self.route_dict = defaultdict(list)
        self.order_dict = {}
        self.ports_dict = {}
        self.vessels_dict = defaultdict(float)

        route_csv = os.path.join(case_folder, "route.csv")

        self._read_route(route_csv)

        order_csv = os.path.join(case_folder, "order.csv")

        self._read_order(order_csv)

        ports_csv = os.path.join(case_folder, "ports.csv")

        self._read_ports(ports_csv)

        vessel_csv = os.path.join(case_folder, "vessels.csv")

        self._read_vessels(vessel_csv)

        self.route_length = defaultdict(lambda: 3)

        self.port_name_to_id = {i: i for i in range(PORT_NUM)}
        self.vessel_name_to_id = {i: i for i in range(VESSEL_NUM)}

    @property
    def node_mapping(self):
        return {
            "static": {
                0: "p1",
                1: "p2",
                2: "p3",
            },
            "dynamic": {
                0: "v1",
                1: "v2"
            }
        }

    @property
    def vessel_stops(self):
        ret = []

        for i in range(VESSEL_NUM):
            ret.append(self.route_dict[i])

        return ret

    def generate_orders(self, tick: int, available_containers: int):
        result = []

        if tick in self.order_dict:
            print(f"got order on tick {tick}")
            for src, dest_dict in self.order_dict[tick].items():
                for dest, qty in self.order_dict[tick][src].items():
                    result.append(Order(tick, src, dest, qty))

        return result

    @property
    def port_initial_info(self):
        cap_list = []
        mt_list = []

        for i in range(PORT_NUM):
            if i in self.ports_dict:
                cap_list.append(self.ports_dict[i]["cap"])
                mt_list.append(self.ports_dict[i]["cntr"])
            else:
                cap_list.append(0)
                mt_list.append(0)

        return {
            "capacity": cap_list,
            "empty": mt_list
        }

    @property
    def vessel_initial_info(self):
        cap_list = []
        mt_list = []
        for i in range(VESSEL_NUM):
            if i in self.vessels_dict:
                cap_list.append(self.vessels_dict[i]['cap'])
                mt_list.append(self.vessels_dict[i]['cntr'])
            else:
                cap_list.append(0)
                mt_list.append(0)

        return {
            "capacity": cap_list,
            "empty": mt_list,
            "route": [0] * VESSEL_NUM
        }

    def get_stop_from_idx(self, vessel_id: int, index: int) -> Stop:
        return self.route_dict[vessel_id][index]

    @property
    def total_containers(self) -> int:
        return TOTAL_CONTAINER_NUM

    @property
    def container_volume(self) -> float:
        return CONTAINER_VOLUME

    @property
    def stop_number(self) -> int:
        return PAST_STOP_NUM, FUTURE_STOP_NUM

    @property
    def vessel_num(self):
        return VESSEL_NUM

    @property
    def port_num(self):
        return PORT_NUM

    def get_reachable_stops(self, vessel_idx, route_idx, next_loc_idx):
        return [(stop.port_idx, stop.arrive_tick) for stop in
                self.route_dict[vessel_idx][next_loc_idx + 1:next_loc_idx + 1 + self.route_length[route_idx]]]

    def get_full_buffer_tick(self, port_name):
        return 0

    def get_empty_buffer_tick(self, port_name):
        return 1

    def get_planed_stops(self, vessel_idx: int, route_idx: int, next_loc_idx: int):
        return [
            (0, 11),
            (2, 13)
        ]

    def get_stop_list(self, *args, **kwargs):
        return [None, Stop(0, 2, 0)], [Stop(4, 6, 2), Stop(10, 12, 3), Stop(20, 22, 4)]

    def _read_ports(self, path: str):
        if os.path.exists(path):
            with open(path, "r") as fp:
                reader = csv.reader(fp)

                next(reader)

                for l in reader:
                    port_id = int(l[0])
                    cap = float(l[1])
                    cntr = int(l[2])

                    if port_id not in self.ports_dict:
                        self.ports_dict[port_id] = {}

                    self.ports_dict[port_id]["cap"] = cap
                    self.ports_dict[port_id]["cntr"] = cntr

    def _read_vessels(self, path: str):
        if os.path.exists(path):
            with open(path, "r") as fp:
                reader = csv.reader(fp)
                next(reader)

                for l in reader:
                    vessel_id = int(l[0])
                    cap = float(l[1])
                    cntr = int(l[2])

                    if vessel_id not in self.vessels_dict:
                        self.vessels_dict[vessel_id] = {}

                    self.vessels_dict[vessel_id]['cap'] = cap
                    self.vessels_dict[vessel_id]['cntr'] = cntr

    def _read_route(self, path: str):
        with open(path, "r") as fp:
            reader = csv.reader(fp)

            next(reader)  # skip header

            for l in reader:
                self.route_dict[int(l[0])].append(
                    Stop(int(l[1]), int(l[2]), int(l[3])))

    def _read_order(self, path: str):
        with open(path, "r") as fp:
            reader = csv.reader(fp)

            next(reader)  # skip header

            for l in reader:
                if l == "":
                    continue

                tick = int(l[0])
                src = int(l[1])
                dest = int(l[2])
                qty = int(l[3])

                if tick not in self.order_dict:
                    self.order_dict[tick] = {}

                if src not in self.order_dict[tick]:
                    self.order_dict[tick][src] = {}

                if dest not in self.order_dict[tick][src]:
                    self.order_dict[tick][src][dest] = {}

                self.order_dict[tick][src][dest] = qty
