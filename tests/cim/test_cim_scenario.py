# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os
import unittest

from maro.event_buffer import EventBuffer, EventState
from maro.simulator.scenarios.cim.business_engine import CimBusinessEngine, CimEventType
from tests.utils import next_step

from .mock_data_container import MockDataContainer

MAX_TICK = 20

def setup_case(case_name: str):
    eb = EventBuffer()
    case_folder = os.path.join("tests", "data", "cim", case_name)

    CimBusinessEngine.__init__ = mock_cim_init_func

    be = CimBusinessEngine(eb, case_folder, MAX_TICK)

    return eb, be


def mock_cim_init_func(self, event_buffer, topology_path, max_tick):

    self._start_tick = 0
    self._max_tick = max_tick
    self._topology_path = topology_path
    self._event_buffer = event_buffer
    self._max_snapshots = None
    self._snapshot_resolution = 1
    
    self._data_cntr = MockDataContainer(topology_path)

    self._vessels = []
    self._ports = []
    self._frame = None

    self._init_frame()

    self._snapshots = self._frame.snapshots

    self._register_events()

    self._load_departure_events()

class TestCimScenarios(unittest.TestCase):
    def setUp(self):
        pass

    def test_init_state(self):
        eb: EventBuffer = None
        be: CimBusinessEngine = None
        eb, be = setup_case("case_01")

        # check frame
        self.assertEqual(3, len(be.frame.ports), "static node number should be same with port number after "
                                                         "initialization")
        self.assertEqual(2, len(be.frame.vessels), "dynamic node number should be same with vessel number "
                                                          "after initialization")

        # check snapshot
        self.assertEqual(MAX_TICK, len(be.snapshots), f"snapshots should be {MAX_TICK} after initialization")

    def test_vessel_moving_correct(self):
        eb, be = setup_case("case_01")
        tick = 0

        #####################################
        # STEP : beginning
        v = be._vessels[0]

        self.assertEqual(0, v.next_loc_idx, "next_loc_idx of vessel 0 should be 0 at beginning")
        self.assertEqual(0, v.last_loc_idx, "last_loc_idx of vessel 0 should be 0 at beginning")

        stop = be._data_cntr.vessel_stops[0, v.next_loc_idx]

        self.assertEqual(0, stop.port_idx, "vessel 0 should parking at port 0 at beginning")

        v = be._vessels[1]

        self.assertEqual(0, v.next_loc_idx, "next_loc_idx of vessel 1 should be 0 at beginning")
        self.assertEqual(0, v.last_loc_idx, "last_loc_idx of vessel 1 should be 0 at beginning")

        stop = be._data_cntr.vessel_stops[1, v.next_loc_idx]

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

        v = be.snapshots["matrices"][2::"vessel_plans"]
        
        # since we already fixed the vessel plans, we just check the value
        for i in range(2):
            self.assertEqual(11, v[i*3+0])
            self.assertEqual(-1, v[i*3+1])
            self.assertEqual(13, v[i*3+2])

        #####################################
        # STEP : tick = 8
        for i in range(6):
            next_step(eb, be, tick)

            tick += 1

        v = be._vessels[0]

        # vessel 0 parking
        self.assertEqual(1, v.next_loc_idx, "next_loc_idx of vessel 0 should be 1 at tick 8")
        self.assertEqual(1, v.last_loc_idx, "last_loc_idx of vessel 0 should be 1 at tick 8")

        stop = be._data_cntr.vessel_stops[0, v.next_loc_idx]

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

        next_step(eb, be, tick)  # move the env to next step, so it will take snapshot for current tick 11

        # we have hard coded the future stops, here we just check if the value correct at each tick
        for i in range(tick - 1):
            # check if the future stop at tick 8 (vessel 0 arrive at port 1)
            stop_list = be.snapshots["vessels"][i:0:["past_stop_list", "past_stop_tick_list"]]

            self.assertEqual(-1, stop_list[0])
            self.assertEqual(-1, stop_list[2])

            stop_list = be.snapshots["vessels"][i:0:["future_stop_list", "future_stop_tick_list"]]

            self.assertEqual(2, stop_list[0])
            self.assertEqual(3, stop_list[1])
            self.assertEqual(4, stop_list[2])
            self.assertEqual(4, stop_list[3])
            self.assertEqual(10, stop_list[4])
            self.assertEqual(20, stop_list[5])

            # check if statistics data correct
            order_states = be.snapshots["ports"][i:0:["shortage", "acc_shortage", "booking", "acc_booking"]]

            # all the value should be 0 for this case
            self.assertEqual(0, order_states[0], f"shortage of port 0 should be 0 at tick {i}")
            self.assertEqual(0, order_states[1], f"acc_shortage of port 0 should be 0 until tick {i}")
            self.assertEqual(0, order_states[2], f"booking of port 0 should be 0 at tick {i}")
            self.assertEqual(0, order_states[3], f"acc_booking of port 0 should be 0 until tick {i}")

            # check fulfillment
            fulfill_states = be.snapshots["ports"][i:0:["fulfillment", "acc_fulfillment"]]

            self.assertEqual(0, fulfill_states[0], f"fulfillment of port 0 should be 0 at tick {i}")
            self.assertEqual(0, fulfill_states[1], f"acc_fulfillment of port 0 should be 0 until tick {i}")

        v = be.snapshots["matrices"][2:: "vessel_plans"]

        # since we already fixed the vessel plans, we just check the value
        for i in range(2):
            self.assertEqual(11, v[i*3+0])
            self.assertEqual(-1, v[i*3+1])
            self.assertEqual(13, v[i*3+2])

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

        # check if statistics data correct
        order_states = be.snapshots["ports"][7:0:["shortage", "acc_shortage", "booking", "acc_booking"]]

        # all the value should be 0 for this case
        self.assertEqual(1, order_states[0], f"shortage of port 0 should be 0 at tick {i}")
        self.assertEqual(1, order_states[1], f"acc_shortage of port 0 should be 0 until tick {i}")
        self.assertEqual(51, order_states[2], f"booking of port 0 should be 0 at tick {i}")
        self.assertEqual(101, order_states[3], f"acc_booking of port 0 should be 0 until tick {i}")

        # check fulfillment
        fulfill_states = be.snapshots["ports"][7:0:["fulfillment", "acc_fulfillment"]]

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
