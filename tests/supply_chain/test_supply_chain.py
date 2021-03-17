import os
import unittest

import numpy as np

from maro.simulator import Env
from maro.simulator.scenarios.supply_chain import ManufactureAction


def build_env(case_name: str, durations: int):
    case_folder = os.path.join("tests", "data", "supply_chain", case_name)

    # config_path = os.path.join(case_folder, "config.yml")

    env = Env(scenario="supply_chain", topology=case_folder, durations=durations)

    return env


class MyTestCase(unittest.TestCase):
    """
    manufacture unit testing:

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

    def test_manufacture_meet_storage_limitation(self):
        """Test sku3 manufacturing."""
        env = build_env("case_01", 100)

        manufacture_nodes = env.snapshot_list["manufacture"]
        manufacture_number = len(manufacture_nodes)
        features = (
        "id", "facility_id", "manufacturing_number", "production_rate", "product_id", "storage_id", "product_unit_cost"
        )

        # tick 0 passed, no product manufacturing.
        env.step(None)

        # try to find which one is sku3 manufacture unit.
        states = manufacture_nodes[env.frame_index::features].flatten().reshape(manufacture_number, -1).astype(np.int)

        for index, state in enumerate(states):
            # Id of sku3 is 3.
            if state[4] == 3:
                sku3_data_model_index = index
                sku3_manufacture_id = state[0]

        # all the id is greater than 0
        self.assertGreater(sku3_manufacture_id, 0)

        # pass an action to start manufacturing for this tick.
        action = ManufactureAction(sku3_manufacture_id, 1)

        env.step({action.id: action})

        states = manufacture_nodes[env.frame_index:sku3_data_model_index:features].flatten().astype(np.int)

        # Sku3 produce rate is 1 per tick, so manufacturing_number should be 1.
        self.assertEqual(1, states[2])

        # leave the action as none will cause manufacture unit stop manufacturing.
        env.step(None)

        states = manufacture_nodes[env.frame_index:sku3_data_model_index:features].flatten().astype(np.int)

        # so manufacturing_number should be 0
        self.assertEqual(0, states[2])



if __name__ == '__main__':
    unittest.main()
