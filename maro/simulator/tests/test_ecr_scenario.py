# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import unittest
import os
from maro.simulator.frame import FrameAttributeType, Frame, SnapshotList, FrameNodeType
from maro.simulator.scenarios.ecr.port import Port
from maro.simulator.scenarios.ecr.vessel import Vessel
from maro.simulator.event_buffer import EventBuffer, EventState
from maro.simulator.scenarios.ecr.business_engine import EcrBusinessEngine, EcrEventType
from mock_data_generator import MockDataGenerator
from maro.simulator.scenarios.ecr.frame_builder import gen_ecr_frame

MAX_TICK = 20


def next_step(eb: EventBuffer, be: EcrBusinessEngine, tick: int):
    if tick > 0:
        # lets post process last tick first before start a new tick
        be.post_step(tick - 1)

    be.step(tick)

    pending_events = eb.execute(tick)

    if len(pending_events) != 0:
        for evt in pending_events:
            evt.state = EventState.FINISHED

        eb.execute(tick)

    be.snapshots.insert_snapshot(be.frame, tick)


def setup_case(case_name: str):
    eb = EventBuffer()
    case_folder = os.path.join(os.path.split(os.path.realpath(__file__))[0], "data", "ecr", case_name)

    EcrBusinessEngine.__init__ = mock_ecr_init_func

    be = EcrBusinessEngine(eb, case_folder, MAX_TICK)

    return eb, be


def mock_ecr_init_func(self, event_buffer, topology_path, max_tick):
    self._topology_path = topology_path
    self._event_buffer = event_buffer
    self._data_generator = MockDataGenerator(topology_path)

    self._vessels = []
    self._ports = []
    self._frame = None

    self._init_frame()

    self._snapshots = SnapshotList(self._frame, max_tick)

    self._register_events()

    self._load_departure_events()


def mock_frame(port_num, vessel_num, future_stop_number):
    return gen_ecr_frame(port_num, vessel_num, future_stop_number)


class TestEcrScenarios(unittest.TestCase):
    def setUp(self):
        pass

    def test_init_state(self):
        eb: EventBuffer = None
        be: EcrBusinessEngine = None
        eb, be = setup_case("case_01")

        # check frame
        self.assertEqual(3, be.frame.static_node_number, "static node number should be same with port number after "
                                                         "initialization")
        self.assertEqual(2, be.frame.dynamic_node_number, "dynamic node number should be same with vessel number "
                                                          "after initialization")

        # check snapshot
        self.assertEqual(0, len(be.snapshots), "snapshots should be 0 after initialization")

        # matrix should be empty
        for m in ["full_on_ports", "full_on_vessels", "vessel_plans"]:
            v = be.snapshots.matrix[0: "full_on_ports"]

            for vv in v:
                for vvv in vv:
                    self.assertEqual(0, vvv)

    def test_vessel_moving_correct(self):
        eb, be = setup_case("case_01")
        tick = 0

        #####################################
        # STEP : beginning
        v = be._vessels[0]

        self.assertEqual(0, v.next_loc_idx, "next_loc_idx of vessel 0 should be 0 at beginning")
        self.assertEqual(0, v.last_loc_idx, "last_loc_idx of vessel 0 should be 0 at beginning")

        stop = be._data_generator.get_stop_from_idx(0, v.next_loc_idx)

        self.assertEqual(0, stop.port_idx, "vessel 0 should parking at port 0 at beginning")

        v = be._vessels[1]

        self.assertEqual(0, v.next_loc_idx, "next_loc_idx of vessel 1 should be 0 at beginning")
        self.assertEqual(0, v.last_loc_idx, "last_loc_idx of vessel 1 should be 0 at beginning")

        stop = be._data_generator.get_stop_from_idx(1, v.next_loc_idx)

        self.assertEqual(1, stop.port_idx, "vessel 1 should parking at port 1 at beginning")

        #####################################
        # STEP : tick = 2
        for i in range(3):
            next_step(eb, be, tick)

            tick += 1

        v = be._vessels[0]

        # if these 2 idx not equal, then means at sailing state
        self.assertEqual(1, v.next_loc_idx, "next_loc_idx of vessel 0 should be 1 at tick 2")
        self.assertEqual(0, v.last_loc_idx, "last_loc_idx of vessel 0 should be 0 at tick 2")

        v = be._vessels[1]

        self.assertEqual(1, v.next_loc_idx, "next_loc_idx of vessel 1 should be 1 at tick 2")
        self.assertEqual(0, v.last_loc_idx, "last_loc_idx of vessel 1 should be 0 at tick 2")

        v = be.snapshots.matrix[2: "vessel_plans"]

        # since we already fixed the vessel plans, we just check the value
        print(v)
        for i in range(2):
            self.assertEqual(11, v[0][i*3+0])
            self.assertEqual(-1, v[0][i*3+1])
            self.assertEqual(13, v[0][i*3+2])

        #####################################
        # STEP : tick = 8
        for i in range(6):
            next_step(eb, be, tick)

            tick += 1

        v = be._vessels[0]

        # vessel 0 parking
        self.assertEqual(1, v.next_loc_idx, "next_loc_idx of vessel 0 should be 1 at tick 8")
        self.assertEqual(1, v.last_loc_idx, "last_loc_idx of vessel 0 should be 1 at tick 8")

        stop = be._data_generator.get_stop_from_idx(0, v.next_loc_idx)

        self.assertEqual(1, stop.port_idx, "vessel 0 should parking at port 1 at tick 8")

        v = be._vessels[1]

        # vessel 1 sailing
        self.assertEqual(1, v.next_loc_idx, "next_loc_idx of vessel 1 should be 1 at tick 8")
        self.assertEqual(0, v.last_loc_idx, "last_loc_idx of vessel 1 should be 0 at tick 8")

        #####################################
        # STEP : tick = 10
        for i in range(2):
            next_step(eb, be, tick)

            tick += 1

        v = be._vessels[0]

        # vessel 0 parking
        self.assertEqual(1, v.next_loc_idx, "next_loc_idx of vessel 0 should be 1 at tick 10")
        self.assertEqual(1, v.last_loc_idx, "last_loc_idx of vessel 0 should be 1 at tick 10")

        v = be._vessels[1]

        # vessel 1 parking
        self.assertEqual(1, v.next_loc_idx, "next_loc_idx of vessel 1 should be 1 at tick 10")
        self.assertEqual(1, v.last_loc_idx, "last_loc_idx of vessel 1 should be 1 at tick 10")

        #####################################
        # STEP : tick = 11
        for i in range(1):
            next_step(eb, be, tick)

            tick += 1

        v = be._vessels[0]

        # vessel 0 parking
        self.assertEqual(2, v.next_loc_idx, "next_loc_idx of vessel 0 should be 2 at tick 11")
        self.assertEqual(1, v.last_loc_idx, "last_loc_idx of vessel 0 should be 1 at tick 11")

        v = be._vessels[1]

        # vessel 1 parking
        self.assertEqual(1, v.next_loc_idx, "next_loc_idx of vessel 1 should be 1 at tick 11")
        self.assertEqual(1, v.last_loc_idx, "last_loc_idx of vessel 1 should be 1 at tick 11")

        # check snapshot
        self.assertEqual(tick, len(be.snapshots), f"there should be {tick} snapshots")

        next_step(eb, be, tick)  # move the env to next step, so it will take snapshot for current tick 11

        # we have hard coded the future stops, here we just check if the value correct at each tick
        for i in range(tick - 1):
            # check if the future stop at tick 8 (vessel 0 arrive at port 1)
            stop_list = be.snapshots.get_attributes(FrameNodeType.DYNAMIC,
                                                    [i, ],
                                                    [0, ],
                                                    ["past_stop_list", "past_stop_tick_list"],
                                                    [0, 1])
            # check if the stop list value correct
            self.assertEqual(-1, stop_list[0])
            self.assertEqual(-1, stop_list[2])

            stop_list = be.snapshots.get_attributes(FrameNodeType.DYNAMIC,
                                                    [i, ],
                                                    [0, ],
                                                    ["future_stop_list", "future_stop_tick_list"],
                                                    [0, 1, 2])

            self.assertEqual(2, stop_list[0])
            self.assertEqual(3, stop_list[1])
            self.assertEqual(4, stop_list[2])
            self.assertEqual(4, stop_list[3])
            self.assertEqual(10, stop_list[4])
            self.assertEqual(20, stop_list[5])

            # check if statistics data correct
            order_states = be.snapshots.get_attributes(FrameNodeType.STATIC,
                                                       [i],
                                                       [0],
                                                       ["shortage", "acc_shortage", "booking", "acc_booking"],
                                                       [0])

            # all the value should be 0 for this case
            self.assertEqual(0, order_states[0], f"shortage of port 0 should be 0 at tick {i}")
            self.assertEqual(0, order_states[1], f"acc_shortage of port 0 should be 0 until tick {i}")
            self.assertEqual(0, order_states[2], f"booking of port 0 should be 0 at tick {i}")
            self.assertEqual(0, order_states[3], f"acc_booking of port 0 should be 0 until tick {i}")

            # check fulfillment
            fulfill_states = be.snapshots.get_attributes(FrameNodeType.STATIC,
                                                         [i],
                                                         [0],
                                                         ["fulfillment", "acc_fulfillment"],
                                                         [0])

            self.assertEqual(0, fulfill_states[0], f"fulfillment of port 0 should be 0 at tick {i}")
            self.assertEqual(0, fulfill_states[1], f"acc_fulfillment of port 0 should be 0 until tick {i}")

        v = be.snapshots.matrix[2: "vessel_plans"]

        # since we already fixed the vessel plans, we just check the value
        for i in range(2):
            self.assertEqual(11, v[0][i*3+0])
            self.assertEqual(-1, v[0][i*3+1])
            self.assertEqual(13, v[0][i*3+2])

    def test_order_state(self):
        eb, be = setup_case("case_02")
        tick = 0

        p = be._ports[0]

        self.assertEqual(0, p.booking, "port 0 have no booking at beginning")
        self.assertEqual(0, p.shortage, "port 0 have no shortage at beginning")
        self.assertEqual(100, p.empty, "port 0 have 100 empty containers at beginning")

        #####################################
        # STEP : tick = 0
        for i in range(1):
            next_step(eb, be, tick)
            tick += 1

        # there should be 10 order generated at tick 0
        self.assertEqual(10, p.booking, "port 0 should have 10 bookings at tick 0")
        self.assertEqual(0, p.shortage, "port 0 have no shortage at tick 0")
        self.assertEqual(90, p.empty, "port 0 have 90 empty containers at tick 0")

        #####################################
        # STEP : tick = 1
        for i in range(1):
            next_step(eb, be, tick)
            tick += 1

        # we have 0 booking, so no shortage
        self.assertEqual(0, p.booking, "port 0 should have 0 bookings at tick 1")
        self.assertEqual(0, p.shortage, "port 0 have no shortage at tick 1")
        self.assertEqual(90, p.empty, "port 0 have 90 empty containers at tick 1")

        #####################################
        # STEP : tick = 3
        for i in range(2):
            next_step(eb, be, tick)
            tick += 1

        # there is an order that take 40 containers
        self.assertEqual(40, p.booking, "port 0 should have 40 booking at tick 3")
        self.assertEqual(0, p.shortage, "port 0 have no shortage at tick 3")
        self.assertEqual(50, p.empty, "port 0 have 90 empty containers at tick 3")

        #####################################
        # STEP : tick = 7
        for i in range(4):
            next_step(eb, be, tick)
            tick += 1

        # there is an order that take 51 containers
        self.assertEqual(51, p.booking, "port 0 should have 51 booking at tick 7")
        self.assertEqual(1, p.shortage, "port 0 have 1 shortage at tick 7")
        self.assertEqual(0, p.empty, "port 0 have 0 empty containers at tick 7")

        # push the simulator to next tick to update snapshot
        next_step(eb, be, tick)

        # check if there is any container missing
        total_cntr_number = sum([port.empty for port in be._ports]) + \
                            sum([vessel.empty for vessel in be._vessels]) + \
                            sum([port.full for port in be._ports]) + \
                            sum([vessel.full for vessel in be._vessels])

        self.assertEqual(be._data_generator.total_containers, total_cntr_number, "containers number should be changed")

        # check if statistics data correct
        order_states = be.snapshots.get_attributes(FrameNodeType.STATIC,
                                                   [7],
                                                   [0],
                                                   ["shortage", "acc_shortage", "booking", "acc_booking"],
                                                   [0])

        # all the value should be 0 for this case
        self.assertEqual(1, order_states[0], f"shortage of port 0 should be 0 at tick {i}")
        self.assertEqual(1, order_states[1], f"acc_shortage of port 0 should be 0 until tick {i}")
        self.assertEqual(51, order_states[2], f"booking of port 0 should be 0 at tick {i}")
        self.assertEqual(101, order_states[3], f"acc_booking of port 0 should be 0 until tick {i}")

        # check fulfillment
        fulfill_states = be.snapshots.get_attributes(FrameNodeType.STATIC,
                                                     [7],
                                                     [0],
                                                     ["fulfillment", "acc_fulfillment"],
                                                     [0])

        self.assertEqual(50, fulfill_states[0], f"fulfillment of port 0 should be 50 at tick {i}")
        self.assertEqual(100, fulfill_states[1], f"acc_fulfillment of port 0 should be 100 until tick {i}")

    def test_order_load_discharge_state(self):
        eb, be = setup_case("case_03")
        tick = 0

        #####################################
        # STEP : tick = 5
        for i in range(6):
            next_step(eb, be, tick)
            tick += 1

        # check if we have load all 50 full container
        p = be._ports[0]
        v = be._vessels[0]

        self.assertEqual(0, p.full, "port 0 should have no full at tick 5")
        self.assertEqual(50, v.full, "all 50 full container should be loaded on vessel 0")
        self.assertEqual(50, p.empty, "remaining empty should be 50 after order generated at tick 5")
        self.assertEqual(0, p.shortage, "no shortage at tick 5 for port 0")
        self.assertEqual(0, p.booking, "no booking at tick 5 for pot 0")

        #####################################
        # STEP : tick = 10
        for i in range(5):
            next_step(eb, be, tick)
            tick += 1

        # at tick 10 vessel 0 arrive at port 1, it should discharge all the full containers
        p1 = be._ports[1]

        self.assertEqual(0, v.full, "all 0 full container on vessel 0 after arrive at port 1 at tick 10")
        self.assertEqual(50, p1.on_consignee,
                         "there should be 50 full containers pending to be empty at tick 10 after discharge")
        self.assertEqual(0, p1.empty, "no empty for port 1 at tick 10")
        self.assertEqual(0, p1.full, "no full for port 1 at tick 10")

        #####################################
        # STEP : tick = 12
        for i in range(2):
            next_step(eb, be, tick)
            tick += 1

        # we hard coded the buffer time to 2, so
        self.assertEqual(0, p1.on_consignee, "all the full become empty at tick 12 for port 1")
        self.assertEqual(50, p1.empty, "there will be 50 empty at tick 12 for port 1")

        #####################################
        # STEP : tick = 13
        next_step(eb, be, tick)

        # check if there is any container missing
        total_cntr_number = sum([port.empty for port in be._ports]) + \
                            sum([vessel.empty for vessel in be._vessels]) + \
                            sum([port.full for port in be._ports]) + \
                            sum([vessel.full for vessel in be._vessels])
        self.assertEqual(be._data_generator.total_containers, total_cntr_number, "containers number should be changed")

    def test_early_discharge(self):
        eb, be = setup_case("case_04")
        tick = 0

        p0 = be._ports[0]
        p1 = be._ports[1]
        p2 = be._ports[2]
        v = be._vessels[0]

        #####################################
        # STEP : tick = 10
        for i in range(11):
            next_step(eb, be, tick)
            tick += 1

        # at tick 10, vessel 0 arrive port 2, it already loaded 50 full, it need to load 50 at port 2, so it will early dicharge 10 empty
        self.assertEqual(0, v.empty, "vessel 0 should early discharge all the empty at tick 10")
        self.assertEqual(100, v.full, "vessel 0 should have 100 full on-board at tick 10")
        self.assertEqual(10, p2.empty, "port 2 have 10 more empty due to early discharge at tick 10")
        self.assertEqual(0, p2.full, "no full at port 2 at tick 10")

        # check if there is any container missing
        total_cntr_number = sum([port.empty for port in be._ports]) \
                            + sum([vessel.empty for vessel in be._vessels]) \
                            + sum([port.full for port in be._ports]) \
                            + sum([vessel.full for vessel in be._vessels]) \
                            - 10  # have add 10 additional empty on vessel initialization, lets reduce it

        self.assertEqual(be._data_generator.total_containers, total_cntr_number,
                         "containers number should not be changed")

        #####################################
        # STEP : tick = 18
        for i in range(8):
            next_step(eb, be, tick)
            tick += 1

        # at tick 18, vessel 0 arrive at port 1, it will discharge all the full
        self.assertEqual(0, v.empty, "vessel 0 should have no empty at tick 18")
        self.assertEqual(0, v.full, "vessel 0 should discharge all full on-board at tick 18")
        self.assertEqual(100, p1.on_consignee, "100 full pending to become empty at port 1 at tick 18")
        self.assertEqual(0, p1.empty, "no empty for port 1 at tick 18")

        #####################################
        # STEP : tick = 20
        for i in range(2):
            next_step(eb, be, tick)
            tick += 1

        self.assertEqual(100, p1.empty, "there should be 100 empty at tick 20 at port 1")


if __name__ == "__main__":
    unittest.main()
