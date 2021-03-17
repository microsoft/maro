import os
import unittest

from maro.simulator import Env


def build_env(case_name: str, durations: int):
    case_folder = os.path.join("tests", "data", "supply_chain", case_name)

    # config_path = os.path.join(case_folder, "config.yml")

    env = Env(scenario="supply_chain", topology=case_folder, durations=durations)

    return env


class MyTestCase(unittest.TestCase):
    def test_manufacture_unit_only(self):
        """
        Test if manufacture unit works as expect, like:

        1. with input sku
            . meet the storage limitation
            . not meet the storage limitation
            . with enough source sku
            . without enough source sku
            . with product rate
            . without product rate
        2. without input sku
            . meet the storage limitation
            . not meet the storage limitation
            . with product rate
            . without product rate

        """

        env = build_env("case_01", 100)

        is_done = False

        while not is_done:
            _, _, is_done = env.step(None)


if __name__ == '__main__':
    unittest.main()
