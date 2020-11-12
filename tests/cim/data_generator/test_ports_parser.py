# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest

from yaml import safe_load

from maro.data_lib.cim.port_parser import PortsParser

default_conf = """
ports:
  demand_port_001:
    capacity: 100000
    empty_return:
      buffer_ticks: 1
      noise: 4
    full_return:
      buffer_ticks: 2
      noise: 5
    initial_container_proportion: 0.5
    order_distribution:
      source:
        noise: 0
        proportion: 0.33
      targets:
        supply_port_001:
          noise: 0
          proportion: 1
  supply_port_001:
    capacity: 1000000
    empty_return:
      buffer_ticks: 3
      noise: 6
    full_return:
      buffer_ticks: 4
      noise: 7
    initial_container_proportion: 0.5
    order_distribution:
      source:
        noise: 0
        proportion: 0.67
"""

class TestPortParser(unittest.TestCase):

    def test_port_parser(self):
        total_cntr = 100

        conf = safe_load(default_conf)

        ppr = PortsParser()

        port_mapping, ports = ppr.parse(conf["ports"], total_cntr)

        # port number should be same with config
        self.assertEqual(2, len(ports))

        # all empty sould be used
        self.assertEqual(total_cntr, sum([p.empty for p in ports]))

        # capacity should same with config
        self.assertListEqual([100000, 1000000], [p.capacity for p in ports])

        # check empty and full return buffer tick
        # and noise
        self.assertListEqual([1, 3], [p.empty_return_buffer.base for p in ports])
        self.assertListEqual([4, 6], [p.empty_return_buffer.noise for p in ports])
        self.assertListEqual([2, 4], [p.full_return_buffer.base for p in ports])
        self.assertListEqual([5, 7], [p.full_return_buffer.noise for p in ports])

        #
        self.assertEqual(len(port_mapping), len(ports))

if __name__ == "__main__":
    unittest.main()
