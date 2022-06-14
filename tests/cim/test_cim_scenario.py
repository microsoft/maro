# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import csv
import os
import pickle
import tempfile
import unittest
from collections import namedtuple
from typing import List, Optional

import yaml

from maro.simulator.utils import random

os.environ["MARO_STREAMIT_ENABLED"] = "true"
os.environ["MARO_STREAMIT_EXPERIMENT_NAME"] = "cim_testing"

from maro.data_lib.cim import dump_from_config
from maro.data_lib.cim.entities import PortSetting, Stop, SyntheticPortSetting, VesselSetting
from maro.data_lib.cim.vessel_stop_wrapper import VesselStopsWrapper
from maro.simulator import Env
from maro.simulator.scenarios.cim.business_engine import CimBusinessEngine
from maro.simulator.scenarios.cim.common import Action, ActionType, DecisionEvent
from maro.simulator.scenarios.cim.ports_order_export import PortOrderExporter

from tests.utils import backends_to_test, compare_dictionary

TOPOLOGY_PATH_CONFIG = "tests/data/cim/case_data/config_folder"
TOPOLOGY_PATH_DUMP = "tests/data/cim/case_data/dump_folder"
TOPOLOGY_PATH_REAL_BIN = "tests/data/cim/case_data/real_folder_bin"
TOPOLOGY_PATH_REAL_CSV = "tests/data/cim/case_data/real_folder_csv"


class TestCimScenarios(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestCimScenarios, self).__init__(*args, **kwargs)

        with open(os.path.join(TOPOLOGY_PATH_CONFIG, "config.yml"), "r") as input_stream:
            self._raw_topology = yaml.safe_load(input_stream)

        self._env: Optional[Env] = None
        self._reload_topology: str = TOPOLOGY_PATH_CONFIG
        self._business_engine: Optional[CimBusinessEngine] = None

        random.clear()

    def _init_env(self, backend_name: str) -> None:
        os.environ["DEFAULT_BACKEND_NAME"] = backend_name
        self._env = Env(
            scenario="cim",
            topology=self._reload_topology,
            start_tick=0,
            durations=200,
            options={"enable-dump-snapshot": tempfile.gettempdir()},
        )
        self._business_engine = self._env.business_engine

    def test_load_from_config(self) -> None:
        for backend_name in backends_to_test:
            self._init_env(backend_name)

            #########################################################
            if len(self._business_engine.configs) > 0:  # Env will not have `configs` if loaded from dump/real.
                self.assertTrue(compare_dictionary(self._business_engine.configs, self._raw_topology))

            self.assertEqual(len(getattr(self._business_engine.frame, "ports")), 22)
            self.assertEqual(self._business_engine._data_cntr.port_number, 22)
            self.assertEqual(len(getattr(self._business_engine.frame, "vessels")), 46)
            self.assertEqual(self._business_engine._data_cntr.vessel_number, 46)
            self.assertEqual(len(self._business_engine.snapshots), 0)

            #########################################################
            # Vessel
            vessels: List[VesselSetting] = self._business_engine._data_cntr.vessels
            for i, vessel in enumerate(vessels):
                vessel_config = self._raw_topology["vessels"][vessel.name]
                self.assertEqual(vessel.index, i)
                self.assertEqual(vessel.capacity, vessel_config["capacity"])
                self.assertEqual(vessel.parking_duration, vessel_config["parking"]["duration"])
                self.assertEqual(vessel.parking_noise, vessel_config["parking"]["noise"])
                self.assertEqual(vessel.start_port_name, vessel_config["route"]["initial_port_name"])
                self.assertEqual(vessel.route_name, vessel_config["route"]["route_name"])
                self.assertEqual(vessel.sailing_noise, vessel_config["sailing"]["noise"])
                self.assertEqual(vessel.sailing_speed, vessel_config["sailing"]["speed"])

            for name, idx in self._business_engine.get_node_mapping()["vessels"].items():
                self.assertEqual(vessels[idx].name, name)

            #########################################################
            # Port
            ports: List[PortSetting] = self._business_engine._data_cntr.ports
            port_names = [port.name for port in ports]
            for i, port in enumerate(ports):
                assert isinstance(port, SyntheticPortSetting)
                port_config = self._raw_topology["ports"][port.name]
                self.assertEqual(port.index, i)
                self.assertEqual(port.capacity, port_config["capacity"])
                self.assertEqual(port.empty_return_buffer.noise, port_config["empty_return"]["noise"])
                self.assertEqual(port.full_return_buffer.noise, port_config["full_return"]["noise"])
                self.assertEqual(port.source_proportion.noise, port_config["order_distribution"]["source"]["noise"])
                for target in port.target_proportions:
                    self.assertEqual(
                        target.noise,
                        port_config["order_distribution"]["targets"][port_names[target.index]]["noise"],
                    )

            for name, idx in self._business_engine.get_node_mapping()["ports"].items():
                self.assertEqual(ports[idx].name, name)

    def test_load_from_real(self) -> None:
        for topology in [TOPOLOGY_PATH_REAL_BIN, TOPOLOGY_PATH_REAL_CSV]:
            self._reload_topology = topology
            for backend_name in backends_to_test:
                self._init_env(backend_name)

                for i, port in enumerate(self._business_engine._ports):
                    self.assertEqual(port.booking, 0)
                    self.assertEqual(port.shortage, 0)

                hard_coded_truth = [556, 0, 20751], [1042, 0, 17320], [0, 0, 25000], [0, 0, 25000]

                self._env.step(action=None)
                for i, port in enumerate(self._business_engine._ports):
                    self.assertEqual(port.booking, hard_coded_truth[i][0])
                    self.assertEqual(port.shortage, hard_coded_truth[i][1])
                    self.assertEqual(port.empty, hard_coded_truth[i][2])

                self._env.reset(keep_seed=True)
                self._env.step(action=None)
                for i, port in enumerate(self._business_engine._ports):
                    self.assertEqual(port.booking, hard_coded_truth[i][0])
                    self.assertEqual(port.shortage, hard_coded_truth[i][1])
                    self.assertEqual(port.empty, hard_coded_truth[i][2])

        self._reload_topology = TOPOLOGY_PATH_CONFIG

    def test_dump_and_load(self) -> None:
        dump_from_config(os.path.join(TOPOLOGY_PATH_CONFIG, "config.yml"), TOPOLOGY_PATH_DUMP, 200)

        self._reload_topology = TOPOLOGY_PATH_DUMP

        # The reloaded Env should have same behaviors
        self.test_load_from_config()
        self.test_vessel_movement()
        self.test_order_state()
        self.test_order_export()
        self.test_early_discharge()

        self._reload_topology = TOPOLOGY_PATH_CONFIG

    def test_vessel_movement(self) -> None:
        for backend_name in backends_to_test:
            self._init_env(backend_name)

            hard_coded_period = [
                67,
                75,
                84,
                67,
                53,
                58,
                51,
                58,
                61,
                49,
                164,
                182,
                146,
                164,
                182,
                146,
                90,
                98,
                79,
                95,
                104,
                84,
                87,
                97,
                78,
                154,
                169,
                136,
                154,
                169,
                94,
                105,
                117,
                94,
                189,
                210,
                167,
                189,
                210,
                167,
                141,
                158,
                125,
                141,
                158,
                125,
            ]
            self.assertListEqual(self._business_engine._data_cntr.vessel_period, hard_coded_period)

            ports: List[PortSetting] = self._business_engine._data_cntr.ports
            port_names: List[str] = [port.name for port in ports]
            vessel_stops: VesselStopsWrapper = self._business_engine._data_cntr.vessel_stops
            vessels: List[VesselSetting] = self._business_engine._data_cntr.vessels

            # Test invalid argument
            self.assertIsNone(vessel_stops[None])

            #########################################################
            for i, vessel in enumerate(vessels):
                start_port_index = port_names.index(vessel.start_port_name)
                self.assertEqual(vessel_stops[i, 0].port_idx, start_port_index)

            #########################################################
            for i, vessel in enumerate(vessels):
                stop_port_indices = [stop.port_idx for stop in vessel_stops[i]]

                raw_route = self._raw_topology["routes"][vessel.route_name]
                route_stop_names = [stop["port_name"] for stop in raw_route]
                route_stop_indices = [port_names.index(name) for name in route_stop_names]
                start_offset = route_stop_indices.index(port_names.index(vessel.start_port_name))

                for j, stop_port_index in enumerate(stop_port_indices):
                    self.assertEqual(stop_port_index, route_stop_indices[(j + start_offset) % len(route_stop_indices)])

            #########################################################
            # STEP: beginning
            for i, vessel in enumerate(self._business_engine._vessels):
                self.assertEqual(vessel.idx, i)
                self.assertEqual(vessel.next_loc_idx, 0)
                self.assertEqual(vessel.last_loc_idx, 0)

            #########################################################
            self._env.step(action=None)
            self.assertEqual(self._env.tick, 5)  # Vessel 35 will trigger the first arrival event at tick 5
            for i, vessel in enumerate(self._business_engine._vessels):
                if i == 35:
                    self.assertEqual(vessel.next_loc_idx, 1)
                    self.assertEqual(vessel.last_loc_idx, 1)
                else:
                    self.assertEqual(vessel.next_loc_idx, 1)
                    self.assertEqual(vessel.last_loc_idx, 0)

            #########################################################
            self._env.step(action=None)
            self.assertEqual(self._env.tick, 6)  # Vessel 27 will trigger the second arrival event at tick 6
            for i, vessel in enumerate(self._business_engine._vessels):
                if i == 27:  # Vessel 27 just arrives
                    self.assertEqual(vessel.next_loc_idx, 1)
                    self.assertEqual(vessel.last_loc_idx, 1)
                elif i == 35:  # Vessel 35 has already departed
                    self.assertEqual(vessel.next_loc_idx, 2)
                    self.assertEqual(vessel.last_loc_idx, 1)
                else:
                    self.assertEqual(vessel.next_loc_idx, 1)
                    self.assertEqual(vessel.last_loc_idx, 0)

            #########################################################
            while self._env.tick < 100:
                self._env.step(action=None)
            self.assertEqual(self._env.tick, 100)
            for i, vessel in enumerate(self._business_engine._vessels):
                expected_next_loc_idx = expected_last_loc_idx = -1
                for j, stop in enumerate(vessel_stops[i]):
                    if stop.arrival_tick == self._env.tick:
                        expected_next_loc_idx = expected_last_loc_idx = j
                        break
                    if stop.arrival_tick > self._env.tick:
                        expected_next_loc_idx = j
                        expected_last_loc_idx = j - 1
                        break

                self.assertEqual(vessel.next_loc_idx, expected_next_loc_idx)
                self.assertEqual(vessel.last_loc_idx, expected_last_loc_idx)

    def test_order_state(self) -> None:
        for backend_name in backends_to_test:
            self._init_env(backend_name)

            for i, port in enumerate(self._business_engine._ports):
                total_containers = self._raw_topology["total_containers"]
                initial_container_proportion = self._raw_topology["ports"][port.name]["initial_container_proportion"]

                self.assertEqual(port.booking, 0)
                self.assertEqual(port.shortage, 0)
                self.assertEqual(port.empty, int(total_containers * initial_container_proportion))

            #########################################################
            self._env.step(action=None)
            self.assertEqual(self._env.tick, 5)

            hard_coded_truth = [  # Should get same results under default random seed
                [223, 0, 14726],
                [16, 0, 916],
                [18, 0, 917],
                [89, 0, 5516],
                [84, 0, 4613],
                [72, 0, 4603],
                [26, 0, 1374],
                [24, 0, 1378],
                [48, 0, 2756],
                [54, 0, 2760],
                [26, 0, 1379],
                [99, 0, 5534],
                [137, 0, 7340],
                [19, 0, 912],
                [13, 0, 925],
                [107, 0, 6429],
                [136, 0, 9164],
                [64, 0, 3680],
                [24, 0, 1377],
                [31, 0, 1840],
                [109, 0, 6454],
                [131, 0, 7351],
            ]
            for i, port in enumerate(self._business_engine._ports):
                self.assertEqual(port.booking, hard_coded_truth[i][0])
                self.assertEqual(port.shortage, hard_coded_truth[i][1])
                self.assertEqual(port.empty, hard_coded_truth[i][2])

    def test_keep_seed(self) -> None:
        for backend_name in backends_to_test:
            self._init_env(backend_name)

            vessel_stops_1: List[List[Stop]] = self._business_engine._data_cntr.vessel_stops
            self._env.step(action=None)
            port_info_1 = [(port.booking, port.shortage, port.empty) for port in self._business_engine._ports]

            self._env.reset(keep_seed=True)
            vessel_stops_2: List[List[Stop]] = self._business_engine._data_cntr.vessel_stops
            self._env.step(action=None)
            port_info_2 = [(port.booking, port.shortage, port.empty) for port in self._business_engine._ports]

            self._env.reset(keep_seed=False)
            vessel_stops_3: List[List[Stop]] = self._business_engine._data_cntr.vessel_stops
            self._env.step(action=None)
            port_info_3 = [(port.booking, port.shortage, port.empty) for port in self._business_engine._ports]

            # Vessel
            for i in range(self._business_engine._data_cntr.vessel_number):
                # 1 and 2 should be totally equal
                self.assertListEqual(vessel_stops_1[i], vessel_stops_2[i])

                # 1 and 3 should have difference
                flag = True
                for stop1, stop3 in zip(vessel_stops_1[i], vessel_stops_3[i]):
                    self.assertListEqual(
                        [stop1.index, stop1.port_idx, stop1.vessel_idx],
                        [stop3.index, stop3.port_idx, stop3.vessel_idx],
                    )
                    if (stop1.arrival_tick, stop1.leave_tick) != (stop3.arrival_tick, stop3.leave_tick):
                        flag = False
                self.assertFalse(flag)

            # Port
            self.assertListEqual(port_info_1, port_info_2)
            self.assertFalse(all(port1 == port3 for port1, port3 in zip(port_info_1, port_info_3)))

    def test_order_export(self) -> None:
        """order.tick, order.src_port_idx, order.dest_port_idx, order.quantity"""
        Order = namedtuple("Order", ["tick", "src_port_idx", "dest_port_idx", "quantity"])

        #
        for enabled in [False, True]:
            exporter = PortOrderExporter(enabled)

            for i in range(5):
                exporter.add(Order(0, 0, 1, i + 1))

            out_folder = tempfile.gettempdir()
            if os.path.exists(f"{out_folder}/orders.csv"):
                os.remove(f"{out_folder}/orders.csv")

            exporter.dump(out_folder)

            if enabled:
                with open(f"{out_folder}/orders.csv") as fp:
                    reader = csv.DictReader(fp)
                    row = 0
                    for line in reader:
                        self.assertEqual(row + 1, int(line["quantity"]))
                        row += 1
            else:  # Should done nothing
                self.assertFalse(os.path.exists(f"{out_folder}/orders.csv"))

    def test_early_discharge(self) -> None:
        for backend_name in backends_to_test:
            self._init_env(backend_name)

            metric, decision_event, is_done = self._env.step(None)
            assert isinstance(decision_event, DecisionEvent)

            self.assertEqual(decision_event.action_scope.load, 1240)
            self.assertEqual(decision_event.action_scope.discharge, 0)
            self.assertEqual(decision_event.early_discharge, 0)

            decision_event = pickle.loads(pickle.dumps(decision_event))  # Test serialization

            load_action = Action(
                vessel_idx=decision_event.vessel_idx,
                port_idx=decision_event.port_idx,
                quantity=1201,
                action_type=ActionType.LOAD,
            )
            discharge_action = Action(
                vessel_idx=decision_event.vessel_idx,
                port_idx=decision_event.port_idx,
                quantity=1,
                action_type=ActionType.DISCHARGE,
            )
            metric, decision_event, is_done = self._env.step([load_action, discharge_action])

            history = []
            while not is_done:
                metric, decision_event, is_done = self._env.step(None)
                assert decision_event is None or isinstance(decision_event, DecisionEvent)
                if decision_event is not None and decision_event.vessel_idx == 35:
                    v = self._business_engine._vessels[35]
                    history.append((v.full, v.empty, v.early_discharge))

            hard_coded_benchmark = [
                (465, 838, 362),
                (756, 547, 291),
                (1261, 42, 505),
                (1303, 0, 42),
                (1303, 0, 0),
                (1303, 0, 0),
                (803, 0, 0),
            ]
            self.assertListEqual(history, hard_coded_benchmark)

            #
            payload_detail_benchmark = {
                "ORDER": ["tick", "src_port_idx", "dest_port_idx", "quantity"],
                "RETURN_FULL": ["src_port_idx", "dest_port_idx", "quantity"],
                "VESSEL_ARRIVAL": ["port_idx", "vessel_idx"],
                "LOAD_FULL": ["port_idx", "vessel_idx"],
                "DISCHARGE_FULL": ["vessel_idx", "port_idx", "from_port_idx", "quantity"],
                "PENDING_DECISION": [
                    "tick",
                    "port_idx",
                    "vessel_idx",
                    "snapshot_list",
                    "action_scope",
                    "early_discharge",
                ],
                "LOAD_EMPTY": ["port_idx", "vessel_idx", "action_type", "quantity"],
                "DISCHARGE_EMPTY": ["port_idx", "vessel_idx", "action_type", "quantity"],
                "VESSEL_DEPARTURE": ["port_idx", "vessel_idx"],
                "RETURN_EMPTY": ["port_idx", "quantity"],
            }
            self.assertTrue(
                compare_dictionary(self._business_engine.get_event_payload_detail(), payload_detail_benchmark),
            )
            port_number = self._business_engine._data_cntr.port_number
            self.assertListEqual(self._business_engine.get_agent_idx_list(), list(range(port_number)))


if __name__ == "__main__":
    unittest.main()
