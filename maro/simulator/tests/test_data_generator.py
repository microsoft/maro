# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os
import unittest

import numpy as np
import matplotlib.pyplot as plt
from yaml import safe_load

from maro.simulator.scenarios.ecr.ecr_data_generator import EcrDataGenerator, GlobalOrderProportion
from maro.simulator.scenarios.ecr.common import Stop

MAX_TICK = 1000


def path_to_case_config(case_name: str):
    return os.path.join(os.path.split(os.path.realpath(__file__))[0],
                        "data", "data_generator", case_name, "config.yml")


def setup_case(max_tick: int, case_name: str):
    return EcrDataGenerator(max_tick, path_to_case_config(case_name))


def load_config(case_name: str):
    with open(path_to_case_config(case_name), "r") as fp:
        return safe_load(fp)


class TestDataGenerator(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_container_fixed_period_distribution(self):
        config = load_config("case_fixed_order_distribution")
        containers = config["total_containers"]
        period = config["container_usage_proportion"]["period"]

        gop = GlobalOrderProportion(config["container_usage_proportion"], MAX_TICK, containers)

        # check period distribution
        self.assertEqual(0, gop.order_period_distribution[0])
        self.assertEqual(0, gop.order_period_distribution[period - 1])
        self.assertEqual(0.5, gop.order_period_distribution[10])

    def test_route(self):
        data_generator = setup_case(MAX_TICK, "case_route")

        expected_stops = data_generator.get_reachable_stops(0, 0, 0)
        self.assertEqual(expected_stops, [(1, 10), (2, 20), (0, 30)])

        expected_stops = data_generator.get_planed_stops(3, 1, 1)
        self.assertEqual(expected_stops, [(4, 20), (0, 30), (3, 40)])

        expected_stops = data_generator.get_stop_list(5, 1, 2)
        self.assertEqual(expected_stops[1][2].arrive_tick, 40)
        self.assertEqual(expected_stops[1][2].port_idx, 0)

        expected_stops = data_generator._vessels._predict_future_stops(5, 1, 2, 3)
        self.assertEqual(expected_stops, [])

    def test_port(self):
        pass

    def test_order(self):
        pass
