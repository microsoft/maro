# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest

import yaml

from maro.data_lib.cim.entities import RoutePoint
from maro.data_lib.cim.route_parser import RoutesParser

conf_str = """
routes:
  route_001:
  - distance: 60
    port_name: supply_port_001
  - distance: 70
    port_name: demand_port_001
  route_002:
  - distance: 60
    port_name: supply_port_001
  - distance: 70
    port_name: supply_port_002
  - distance: 80
    port_name: demand_port_002
"""

class TestRoutePoint(unittest.TestCase):
    def test_route_parser(self):
        conf = yaml.safe_load(conf_str)

        parser = RoutesParser()

        route_mapping, routes = parser.parse(conf["routes"])

        # there should be 2 routes
        self.assertEqual(2, len(route_mapping))
        self.assertEqual(2, len(routes))

        self.assertEqual(0, route_mapping['route_001'])
        self.assertEqual(1, route_mapping['route_002'])

        # check distance
        self.assertListEqual([60, 70], [r.distance for r in routes[0]])
        self.assertListEqual([60, 70, 80], [r.distance for r in routes[1]])


if __name__=="__main__":
    unittest.main()
