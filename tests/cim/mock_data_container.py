# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import csv
import os
from collections import defaultdict

from maro.data_lib.cim.entities import Order, PortSetting, Stop, VesselSetting

PORT_NUM = 3
VESSEL_NUM = 2
PAST_STOP_NUM = 2
FUTURE_STOP_NUM = 3
TOTAL_CONTAINER_NUM = 100
CONTAINER_VOLUME = 1


class MockVesselStopsWrapper:
    def __init__(self, stops):
        self._stops = stops

    def __getitem__(self, key):
        key_type = type(key)

        if key_type == int:
            # get stops for vessel
            vessel_idx = key
            return self._stops[vessel_idx]
        elif key_type == tuple:
            vessel_idx = key[0]
            loc_idx = key[1]

            return self._stops[vessel_idx][loc_idx]

        else:
            ret = []

            for i in range(VESSEL_NUM):
                ret.append(self._stops[i])

            return ret

class MockEmptyReturnBufferWrapper:
    def __getitem__(self, key):
        return 1


class MockFullReturnBufferWrapper:
    def __getitem__(self, key):
        return 0


class MockVesselSailingPlanWrapper:
    def __getitem__(self, key):
        return [
            (0, 11),
            (2, 13)
        ]


class MockVeselPastStopWapper:
    def __getitem__(self, key):
        return [None, Stop(-1, 0, 2, 0, 0)]


class MockVesselFutureStopWrapper:
    def __getitem__(self, key):
        return [Stop(-1, 4, 6, 2, 0), Stop(-1, 10, 12, 3, 0), Stop(-1, 20, 22, 4, 0)]


class MockReachableStopsWrapper:
    def __init__(self, stops, route_length):
        self._stops = stops
        self._route_length = route_length

    def __getitem__(self, key):
        vessel_idx = key[0]
        route_idx = key[1]
        next_loc_idx = key[2]

        return [(stop.port_idx, stop.arrive_tick) for stop in
                    self._stops[vessel_idx][next_loc_idx + 1:next_loc_idx + 1 + self._route_length[route_idx]]]

class MockDataContainer:
    def __init__(self, case_folder: str):
        self.route_dict = defaultdict(list)
        self.order_dict = {}
        self.ports_dict = {}
        self.vessels_dict = defaultdict(float)
        self._ports = []
        self._vessels = []

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



        self._vessel_stops_wrapper = MockVesselStopsWrapper(self.route_dict)
        self._reachable_stops_wrapper = MockReachableStopsWrapper(self.route_dict, self.route_length)

    @property
    def past_stop_number(self) -> int:
        return PAST_STOP_NUM

    @property
    def future_stop_number(self) -> int:
        return FUTURE_STOP_NUM

    @property
    def ports(self):
        return self._ports

    @property
    def port_number(self) -> int:
        return PORT_NUM

    @property
    def vessels(self):
        return self._vessels

    @property
    def vessel_number(self) -> int:
        return VESSEL_NUM

    @property
    def container_volume(self):
        return CONTAINER_VOLUME

    @property
    def vessel_stops(self) :
        return self._vessel_stops_wrapper

    @property
    def empty_return_buffers(self):
        return MockEmptyReturnBufferWrapper()

    @property
    def full_return_buffers(self):
        return MockFullReturnBufferWrapper()

    @property
    def vessel_past_stops(self):
        return MockVeselPastStopWapper()

    @property
    def vessel_future_stops(self):
        return MockVesselFutureStopWrapper()

    @property
    def vessel_planned_stops(self):
        return MockVesselSailingPlanWrapper()

    @property
    def reachable_stops(self):
        return self._reachable_stops_wrapper

    @property
    def vessel_peroid(self):
        pass

    @property
    def route_mapping(self):
        return {"r1": 0}

    @property
    def vessel_mapping(self):
        return {v.name: v.index for v in self._vessels}

    @property
    def port_mapping(self):
        return {p.name: p.index for p in self._ports}

    def get_orders(self, tick: int, total_empty_container: int):
        result = []

        if tick in self.order_dict:

            for src, _ in self.order_dict[tick].items():
                for dest, qty in self.order_dict[tick][src].items():
                    result.append(Order(tick, src, dest, qty))

        return result

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

                    port = PortSetting(port_id, f"p{port_id}", cap, cntr, None, None, None, None)

                    self._ports.append(port)

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

                    vessel = VesselSetting(vessel_id, f"v{vessel_id}", cap, "r1", None, None, None, None, None, cntr)

                    self._vessels.append(vessel)

    def _read_route(self, path: str):
        with open(path, "r") as fp:
            reader = csv.reader(fp)

            next(reader)  # skip header

            for l in reader:
                self.route_dict[int(l[0])].append(
                    Stop(-1, int(l[1]), int(l[2]), int(l[3]), int(l[0])))

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
