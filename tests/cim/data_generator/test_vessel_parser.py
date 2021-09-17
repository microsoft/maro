# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest

import yaml

from maro.data_lib.cim.parsers import parse_vessels

conf_str = """
vessels:
  rt1_vessel_001:
    capacity: 92400
    parking:
      duration: 1
      noise: 0
    route:
      initial_port_name: supply_port_001
      route_name: route_001
    sailing:
      noise: 0
      speed: 10
  rt1_vessel_002:
    capacity: 92400
    parking:
      duration: 1
      noise: 0
    route:
      initial_port_name: demand_port_001
      route_name: route_001
    sailing:
      noise: 0
      speed: 10
"""


class TestVesselParser(unittest.TestCase):
    def test_vessel_parse(self):
        conf = yaml.safe_load(conf_str)

        vessel_mapping, vessels = parse_vessels(conf["vessels"])

        self.assertEqual(2, len(vessel_mapping))
        self.assertEqual(2, len(vessels))

        self.assertEqual("rt1_vessel_001", vessels[0].name)
        self.assertEqual("rt1_vessel_002", vessels[1].name)

        # check capacity
        self.assertListEqual([92400, 92400], [v.capacity for v in vessels])

        self.assertListEqual([1, 1], [v.parking_duration for v in vessels])
        self.assertListEqual([0, 0], [v.parking_noise for v in vessels])
        self.assertListEqual([10, 10], [v.sailing_speed for v in vessels])
        self.assertListEqual([0, 0], [v.sailing_noise for v in vessels])


if __name__ == "__main__":
    unittest.main()
