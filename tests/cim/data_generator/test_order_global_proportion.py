# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest
from math import floor

from maro.data_lib.cim.parsers import parse_global_order_proportion


class TestOrderGlobalProportion(unittest.TestCase):
    def test_orders_without_noise(self):
        total_cnt = 100
        max_tick = 50
        period = 10
        ratio = 0.02

        conf = {
            "period": period,
            "sample_nodes": [
                (0, ratio),
                (period - 1, ratio),
            ],
            "sample_noise": 0,
        }

        prop = parse_global_order_proportion(conf, total_container=total_cnt, max_tick=max_tick)

        # check the order number
        self.assertEqual(floor(total_cnt * ratio) * max_tick, prop.sum())

        # check proportion, it should be a line
        self.assertListEqual([floor(total_cnt * ratio)] * max_tick, list(prop))

    def test_orders_with_noise(self):
        total_cnt = 100
        max_tick = 50
        period = 10
        ratio = 0.02

        conf = {
            "period": period,
            "sample_nodes": [
                (0, ratio),
                (period - 1, ratio),
            ],
            "sample_noise": 0.1,
        }

        prop = parse_global_order_proportion(conf, total_container=total_cnt, max_tick=max_tick)

        self.assertTrue(prop.sum() > 0)

    def test_orders_with_start_tick(self):
        total_cnt = 100
        start_tick = 10
        max_tick = 50
        period = 10
        ratio = 0.02

        conf = {
            "period": period,
            "sample_nodes": [
                (0, ratio),
                (period - 1, ratio),
            ],
            "sample_noise": 0,
        }

        prop = parse_global_order_proportion(conf, total_container=total_cnt, start_tick=start_tick, max_tick=max_tick)

        self.assertEqual(floor(total_cnt * ratio) * (max_tick - start_tick), prop.sum())


if __name__ == "__main__":
    unittest.main()
