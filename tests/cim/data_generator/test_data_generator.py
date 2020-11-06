# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import unittest

import yaml

from maro.data_lib.cim.cim_data_generator import CimDataGenerator
from maro.data_lib.cim.entities import CimDataCollection

MAX_TICK = 20

class TestDataGenerator(unittest.TestCase):
    def test_data_generator_without_noise(self):
        config_path = os.path.join("tests", "data", "cim", "data_generator", "dumps", "config.yml")

        ge = CimDataGenerator()

        dc: CimDataCollection = ge.gen_data(config_path, max_tick=MAX_TICK)

        self.assertEqual(MAX_TICK, len(dc.order_proportion))
        self.assertEqual(100000, dc.total_containers)

        # no noise
        self.assertListEqual([0.02 * dc.total_containers] * MAX_TICK, list(dc.order_proportion))

        # there should be 4 ports
        self.assertEqual(4, len(dc.ports_settings))

        # and 5 vessels
        self.assertEqual(5, len(dc.vessels_settings))

        # check vessel capacity
        self.assertListEqual([92400, 92400, 187600, 187600, 187600], [v.capacity for v in dc.vessels_settings])
        self.assertListEqual([10] * 5, [v.sailing_speed for v in dc.vessels_settings])
        self.assertListEqual([0] * 5, [v.sailing_noise for v in dc.vessels_settings])
        self.assertListEqual([1] * 5, [v.parking_duration for v in dc.vessels_settings])
        self.assertListEqual([0] * 5, [v.parking_noise for v in dc.vessels_settings])

        # empty of vessel should be 0 by default from configuration
        self.assertListEqual([0] * 5, [v.empty for v in dc.vessels_settings])

        # port
        self.assertListEqual([100000,100000, 1000000, 100000], [p.capacity for p in dc.ports_settings])
        self.assertListEqual([0.25 * dc.total_containers] * 4, [p.empty for p in dc.ports_settings])

        # TODO: more

if __name__=="__main__":
    unittest.main()
