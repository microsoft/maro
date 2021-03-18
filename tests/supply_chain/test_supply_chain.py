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

        storage_nodes = env.snapshot_list["storage"]
        storage_features = ("id", "facility_id", "capacity", "remaining_space", "unit_storage_cost")

        manufacture_nodes = env.snapshot_list["manufacture"]
        manufacture_number = len(manufacture_nodes)
        manufacture_features = (
            "id", "facility_id", "manufacturing_number", "production_rate", "product_id", "storage_id", "product_unit_cost"
        )

        # tick 0 passed, no product manufacturing.
        env.step(None)

        states = manufacture_nodes[env.frame_index::manufacture_features].flatten().reshape(manufacture_number, -1).astype(np.int)

        # try to find which one is sku3 manufacture unit.
        for index, state in enumerate(states):
            # Id of sku3 is 3.
            if state[4] == 3:
                sku3_data_model_index = index
                sku3_manufacture_id = state[0]
                sku3_storage_id = state[5]

        # try to find sku3's storage from env.summary
        sku3_storage_index = env.summary["node_mapping"]["mapping"][sku3_storage_id][1]

        storage_states = storage_nodes[env.frame_index:sku3_storage_index:storage_features].flatten().astype(np.int)

        # there should be 80 units been taken at the beginning according to the config file.
        # so remaining space should be 20
        self.assertEqual(20, storage_states[3])
        # capacity is 100 by config
        self.assertEqual(100, storage_states[2])

        # all the id is greater than 0
        self.assertGreater(sku3_manufacture_id, 0)

        # pass an action to start manufacturing for this tick.
        action = ManufactureAction(sku3_manufacture_id, 1)

        env.step({action.id: action})

        states = manufacture_nodes[env.frame_index:sku3_data_model_index:manufacture_features].flatten().astype(np.int)

        # Sku3 produce rate is 1 per tick, so manufacturing_number should be 1.
        self.assertEqual(1, states[2])

        storage_states = storage_nodes[env.frame_index:sku3_storage_index:storage_features].flatten().astype(np.int)

        # now remaining space should be 19
        self.assertEqual(19, storage_states[3])

        # leave the action as none will cause manufacture unit stop manufacturing.
        env.step(None)

        states = manufacture_nodes[env.frame_index:sku3_data_model_index:manufacture_features].flatten().astype(np.int)

        # so manufacturing_number should be 0
        self.assertEqual(0, states[2])

        # let is generate 20, but actually it can only procedure 19 because the storage will reach the limitation
        env.step({sku3_manufacture_id: ManufactureAction(sku3_manufacture_id, 20)})

        states = manufacture_nodes[env.frame_index:sku3_data_model_index:manufacture_features].flatten().astype(np.int)

        # so manufacture_number should be 19 instead 20
        self.assertEqual(19, states[2])

        storage_states = storage_nodes[env.frame_index:sku3_storage_index:storage_features].flatten().astype(np.int)

        # now remaining space should be 0
        self.assertEqual(0, storage_states[3])

    def test_manufacture_meet_source_lack(self):
        """Test sku4 manufacturing, this sku supplier does not have enough source material at the begging
            , so it cannot produce anything without consumer purchase."""
        env = build_env("case_01", 100)

        storage_nodes = env.snapshot_list["storage"]
        storage_features = ("id", "facility_id", "capacity", "remaining_space", "unit_storage_cost")

        manufacture_nodes = env.snapshot_list["manufacture"]
        manufacture_number = len(manufacture_nodes)
        manufacture_features = (
            "id", "facility_id", "manufacturing_number", "production_rate", "product_id", "storage_id",
            "product_unit_cost"
        )

        # tick 0 passed, no product manufacturing.
        env.step(None)

        states = manufacture_nodes[env.frame_index::manufacture_features].flatten().reshape(manufacture_number,
                                                                                            -1).astype(np.int)

        # try to find which one is sku3 manufacture unit.
        for index, state in enumerate(states):
            # Id of sku4 is 4.
            if state[4] == 4:
                sku4_data_model_index = index
                sku4_manufacture_id = state[0]
                sku4_storage_id = state[5]

        # try to find sku4's storage from env.summary
        sku4_storage_index = env.summary["node_mapping"]["mapping"][sku4_storage_id][1]

        # the storage should be same as initialized (50 + 0).
        storage_states = storage_nodes[env.frame_index:sku4_storage_index:storage_features].flatten().astype(np.int)

        # capacity is same as configured.
        self.assertEqual(200, storage_states[2])

        # remaining space should be capacity - (50+0)
        self.assertEqual(200 - (50+0), storage_states[3])

        # no manufacture number as we have not pass any action
        manufature_states = manufacture_nodes[env.frame_index:sku4_data_model_index:manufacture_features].flatten().astype(np.int)

        # manufacturing_number should be 0
        self.assertEqual(0, manufature_states[2])

        # production rate should be 0
        self.assertEqual(0, manufature_states[3])

        # output product id should be same as configured.
        self.assertEqual(4, manufature_states[4])

        # product unit cost should be same as configured.
        self.assertEqual(4, manufature_states[6])

        # push to the end, the storage should not changed


if __name__ == '__main__':
    unittest.main()
