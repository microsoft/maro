# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import unittest

from maro.data_lib import BinaryConverter
from maro.event_buffer import EventBuffer
from maro.simulator import Env
from maro.simulator.scenarios.citi_bike.business_engine import CitibikeBusinessEngine
from maro.simulator.scenarios.citi_bike.events import CitiBikeEvents
from tests.utils import be_run_to_end, next_step


def setup_case(case_name: str, max_tick:int):
    config_path = os.path.join("tests/data/citi_bike", case_name)

    # enable binary exist

    # trips.bin
    trips_bin = os.path.join(config_path, "trips.bin")

    if not os.path.exists(trips_bin):
        converter = BinaryConverter(trips_bin, os.path.join("tests/data/citi_bike", "trips.meta.yml"))

        converter.add_csv(os.path.join(config_path, "trips.csv"))
        converter.flush()

    # weathers.bin
    weathers_bin = os.path.join("tests/data/citi_bike", "weathers.bin")

    if not os.path.exists(weathers_bin):
        converter = BinaryConverter(weathers_bin, os.path.join("tests/data/citi_bike", "weather.meta.yml"))

        converter.add_csv(os.path.join("tests/data/citi_bike", "weather.csv"))
        converter.flush()

    eb = EventBuffer()
    be = CitibikeBusinessEngine(event_buffer=eb, topology=config_path, start_tick=0, max_tick=max_tick, snapshot_resolution=1, max_snapshots=None, additional_options={})

    return eb, be

class TestCitibike(unittest.TestCase):
    def test_trips_without_shortage(self):
        """Normal case without shortage, case_1"""
        eb, be = setup_case("case_1", max_tick=10)

        next_step(eb, be, 0)

        station_num = len(be.frame.stations)

        station_0 = be.frame.stations[0]
        station_1 = be.frame.stations[1]

        # check bikes at station 0, 1 should be moved
        self.assertEqual(4, station_0.bikes)
        self.assertEqual(10, station_1.bikes)

        pending_evts = eb.get_pending_events(5)

        # check event in pending pool, there should be 1 returned event
        self.assertEqual(1, len(pending_evts))
        self.assertEqual(CitiBikeEvents.ReturnBike, pending_evts[0].event_type)

        next_step(eb, be, 1)

        # station 0 and 1 have 1 trip
        self.assertEqual(3, station_0.bikes)
        self.assertEqual(9, station_1.bikes)

        # no shortage
        self.assertEqual(0, station_0.shortage)
        self.assertEqual(0, station_1.shortage)

        # check if snapshot correct
        states = be.snapshots["stations"][::["shortage", "bikes", "fulfillment", "trip_requirement"]]

        # reshape by tick, attribute numbr and station number
        states = states.reshape(-1, station_num,  4)

        self.assertEqual(2, len(states))

        states_at_tick_0 = states[0]
        states_at_tick_1 = states[1]

        # no shortage
        self.assertEqual(0, states_at_tick_0[:,0].sum())
        self.assertEqual(4+10, states_at_tick_0[:,1].sum())

        # since no shortage, trips == fulfillments
        self.assertEqual(states_at_tick_0[:,2].sum(), states_at_tick_0[:,3].sum())

        #
        self.assertEqual(0, states_at_tick_1[:,0].sum())
        self.assertEqual(3+9, states_at_tick_1[:,1].sum())

        self.assertEqual(states_at_tick_1[:,2].sum(), states_at_tick_1[:,3].sum())

    def test_trips_on_multiple_epsiode(self):
        """Test if total trips of multiple episodes with same config are same"""

        max_ep = 100

        eb, be = setup_case("case_1", max_tick=100)

        total_trips_list = []

        for ep in range(max_ep):
            eb.reset()
            be.reset()

            be_run_to_end(eb, be)

            total_trips = be.snapshots["stations"][::"trip_requirement"].sum()
            shortage_and_fulfillment = be.snapshots["stations"][::["shortage", "fulfillment"]].sum()

            self.assertEqual(total_trips, shortage_and_fulfillment)

            total_trips_list.append(total_trips)

            # if same with previous episodes
            self.assertEqual(total_trips_list[0], total_trips)


    def test_trips_with_shortage(self):
        """Test if shortage states correct"""

        eb, be = setup_case("case_2", max_tick=5)

        stations_snapshots = be.snapshots["stations"]

        be_run_to_end(eb, be)

        states_at_tick_0 = stations_snapshots[0:0:["shortage", "bikes"]]

        shortage_at_tick_0 = states_at_tick_0[0]
        bikes_at_tick_0 = states_at_tick_0[1]

        # there should be no shortage, and 4 left
        self.assertEqual(0, shortage_at_tick_0)
        self.assertEqual(4, bikes_at_tick_0)

        # there should be 6 trips from 1st station, so there will be 2 shortage
        states_at_tick_1 = stations_snapshots[1:0:["shortage", "bikes", "trip_requirement"]]

        self.assertEqual(2, states_at_tick_1[0])
        self.assertEqual(0, states_at_tick_1[1])
        self.assertEqual(6, states_at_tick_1[2])


if __name__ == "__main__":
    unittest.main()
