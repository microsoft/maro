# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest
from math import floor

from maro.data_lib.cim.global_order_proportion import GlobalOrderProportion


class TestOrderGlobalProportion(unittest.TestCase):

    def test_orders_without_noise(self):
        total_cntrs = 100
        max_tick = 50
        period = 10
        ratio = 0.02

        conf ={
            "period": period,
            "sample_nodes": [
                (0, ratio),
                (period-1, ratio)
            ],
            "sample_noise": 0
        }

        op = GlobalOrderProportion()

        prop = op.parse(conf, total_container=total_cntrs, max_tick=max_tick)

        # check the order number
        self.assertEqual(floor(total_cntrs * ratio) * max_tick, prop.sum())

        # check proportion, it should be a line
        self.assertListEqual([floor(total_cntrs * ratio)] * max_tick, list(prop))

    def test_orders_with_noise(self):
        total_cntrs = 100
        max_tick = 50
        period = 10
        ratio = 0.02

        conf ={
            "period": period,
            "sample_nodes": [
                (0, ratio),
                (period-1, ratio)
            ],
            "sample_noise": 0.1
        }

        op = GlobalOrderProportion()

        prop = op.parse(conf, total_container=total_cntrs, max_tick=max_tick)

        self.assertTrue(prop.sum() > 0)

    def test_orders_with_start_tick(self):
        total_cntrs = 100
        start_tick = 10
        max_tick = 50
        period = 10
        ratio = 0.02

        conf ={
            "period": period,
            "sample_nodes": [
                (0, ratio),
                (period-1, ratio)
            ],
            "sample_noise": 0
        }

        op = GlobalOrderProportion()

        prop = op.parse(conf, total_container=total_cntrs, start_tick=start_tick, max_tick=max_tick)

        self.assertEqual(floor(total_cntrs * ratio) * (max_tick - start_tick), prop.sum())

if __name__ == "__main__":
    unittest.main()
