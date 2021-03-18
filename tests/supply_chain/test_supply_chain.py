import os
import unittest

import numpy as np

from maro.simulator import Env
from maro.simulator.scenarios.supply_chain import ManufactureAction
from maro.simulator.scenarios.supply_chain import StorageUnit

def build_env(case_name: str, durations: int):
    case_folder = os.path.join("tests", "data", "supply_chain", case_name)

    # config_path = os.path.join(case_folder, "config.yml")

    env = Env(scenario="supply_chain", topology=case_folder, durations=durations)

    return env


def get_product_dict_from_storage(env: Env, frame_index: int, node_index: int):
    product_list = env.snapshot_list["storage"][frame_index:node_index:"product_list"].flatten().astype(np.int)
    product_number = env.snapshot_list["storage"][frame_index:node_index:"product_number"].flatten().astype(np.int)

    return {pid: pnum for pid, pnum in zip(product_list, product_number)}


SKU1_ID = 1
SKU2_ID = 2
SKU3_ID = 3
SKU4_ID = 4


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
            "id", "facility_id", "manufacturing_number", "production_rate", "product_id", "storage_id",
            "product_unit_cost"
        )

        ############################### TICK: 0 ######################################

        # tick 0 passed, no product manufacturing.
        env.step(None)

        states = manufacture_nodes[env.frame_index::manufacture_features].flatten().reshape(manufacture_number,
                                                                                            -1).astype(np.int)

        # try to find which one is sku3 manufacture unit.
        for index, state in enumerate(states):
            # Id of sku3 is 3.
            if state[4] == SKU3_ID:
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

        product_dict = get_product_dict_from_storage(env, env.frame_index, sku3_storage_index)

        # number should be same as configuration at beginning.
        # 80 sku3
        self.assertEqual(80, product_dict[SKU3_ID])

        # all the id is greater than 0
        self.assertGreater(sku3_manufacture_id, 0)

        ############################### TICK: 1 ######################################

        # pass an action to start manufacturing for this tick.
        action = ManufactureAction(sku3_manufacture_id, 1)

        env.step({action.id: action})

        states = manufacture_nodes[env.frame_index:sku3_data_model_index:manufacture_features].flatten().astype(np.int)

        # Sku3 produce rate is 1 per tick, so manufacturing_number should be 1.
        self.assertEqual(1, states[2])

        storage_states = storage_nodes[env.frame_index:sku3_storage_index:storage_features].flatten().astype(np.int)

        # now remaining space should be 19
        self.assertEqual(19, storage_states[3])

        product_dict = get_product_dict_from_storage(env, env.frame_index, sku3_storage_index)

        # sku3 number should be 80 + 1
        self.assertEqual(80 + 1, product_dict[SKU3_ID])

        ############################### TICK: 2 ######################################

        # leave the action as none will cause manufacture unit stop manufacturing.
        env.step(None)

        states = manufacture_nodes[env.frame_index:sku3_data_model_index:manufacture_features].flatten().astype(np.int)

        # so manufacturing_number should be 0
        self.assertEqual(0, states[2])

        product_dict = get_product_dict_from_storage(env, env.frame_index, sku3_storage_index)

        # sku3 number should be same as last tick
        self.assertEqual(80 + 1, product_dict[SKU3_ID])

        # let is generate 20, but actually it can only procedure 19 because the storage will reach the limitation
        env.step({sku3_manufacture_id: ManufactureAction(sku3_manufacture_id, 20)})

        states = manufacture_nodes[env.frame_index:sku3_data_model_index:manufacture_features].flatten().astype(np.int)

        # so manufacture_number should be 19 instead 20
        self.assertEqual(19, states[2])

        storage_states = storage_nodes[env.frame_index:sku3_storage_index:storage_features].flatten().astype(np.int)

        # now remaining space should be 0
        self.assertEqual(0, storage_states[3])

        product_dict = get_product_dict_from_storage(env, env.frame_index, sku3_storage_index)

        # sku3 number should be 100
        self.assertEqual(80 + 1 + 19, product_dict[SKU3_ID])

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

        ############################### TICK: 0 ######################################

        # tick 0 passed, no product manufacturing.
        env.step(None)

        states = manufacture_nodes[env.frame_index::manufacture_features].flatten().reshape(manufacture_number,
                                                                                            -1).astype(np.int)

        # try to find which one is sku3 manufacture unit.
        for index, state in enumerate(states):
            # Id of sku4 is 4.
            if state[4] == SKU4_ID:
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
        self.assertEqual(200 - (50 + 0), storage_states[3])

        # no manufacture number as we have not pass any action
        manufature_states = manufacture_nodes[
                            env.frame_index:sku4_data_model_index:manufacture_features].flatten().astype(np.int)

        # manufacturing_number should be 0
        self.assertEqual(0, manufature_states[2])

        # production rate should be 0
        self.assertEqual(0, manufature_states[3])

        # output product id should be same as configured.
        self.assertEqual(4, manufature_states[4])

        # product unit cost should be same as configured.
        self.assertEqual(4, manufature_states[6])

        product_dict = get_product_dict_from_storage(env, env.frame_index, sku4_storage_index)

        # 50 sku4 at beginning
        self.assertEqual(50, product_dict[SKU4_ID])

        # 0 sku2
        self.assertEqual(0, product_dict[SKU2_ID])

        ############################### TICK: 1 - end ######################################

        is_done = False

        while not is_done:
            # push to the end, the storage should not changed, no matter what production rate we give it.
            _, _, is_done = env.step({sku4_manufacture_id: ManufactureAction(sku4_manufacture_id, 10)})

        manufature_states = manufacture_nodes[
                            env.frame_index:sku4_data_model_index:manufacture_features].flatten().astype(
            np.int)

        # manufacturing_number should be 0
        self.assertEqual(0, manufature_states[2])

        # production rate should be 10
        self.assertEqual(10, manufature_states[3])

        # output product id should be same as configured.
        self.assertEqual(SKU4_ID, manufature_states[4])

        # product unit cost should be same as configured.
        self.assertEqual(4, manufature_states[6])

        product_dict = get_product_dict_from_storage(env, env.frame_index, sku4_storage_index)

        # 50 sku4 at beginning
        self.assertEqual(50, product_dict[SKU4_ID])

        # 0 sku2
        self.assertEqual(0, product_dict[SKU2_ID])

    def test_manufacture_meet_avg_storage_limitation(self):
        """Test on sku1, it is configured with nearly full initial states."""

        env = build_env("case_01", 100)

        storage_nodes = env.snapshot_list["storage"]
        storage_features = ("id", "facility_id", "capacity", "remaining_space", "unit_storage_cost")

        manufacture_nodes = env.snapshot_list["manufacture"]
        manufacture_number = len(manufacture_nodes)
        manufacture_features = (
            "id", "facility_id", "manufacturing_number", "production_rate", "product_id", "storage_id",
            "product_unit_cost"
        )

        ############################### TICK: 0 ######################################

        # tick 0 passed, no product manufacturing, verified in above case, pass checking it here.
        env.step(None)

        states = manufacture_nodes[env.frame_index::manufacture_features].flatten().reshape(manufacture_number,
                                                                                            -1).astype(np.int)
        # try to find which one is sku3 manufacture unit.
        for index, state in enumerate(states):
            # Id of sku1 is 1.
            if state[4] == SKU1_ID:
                sku1_data_model_index = index
                sku1_manufacture_id = state[0]
                sku1_storage_id = state[5]

        sku1_storage_index = env.summary["node_mapping"]["mapping"][sku1_storage_id][1]

        ############################### TICK: 1 ######################################

        # ask sku1 manufacture start manufacturing, rate is 10.
        env.step({sku1_manufacture_id: ManufactureAction(sku1_storage_index, 10)})

        storage_states = storage_nodes[env.frame_index:sku1_storage_index:storage_features].flatten().astype(np.int)
        manufacture_states = manufacture_nodes[
                             env.frame_index:sku1_data_model_index:manufacture_features].flatten().astype(np.int)

        # we can produce 4 sku1, as it will meet storage avg limitation per sku
        self.assertEqual(4, manufacture_states[2])

        # but the production rate is same as action
        self.assertEqual(10, manufacture_states[3])

        # so storage remaining space should be 200 - ((96 + 4) + (100 - 4*2))
        self.assertEqual(200 - ((96 + 4) + (100 - 4 * 2)), storage_states[3])

        product_dict = get_product_dict_from_storage(env, env.frame_index, sku1_storage_index)

        # number of sku1 should 100, just reach the avg storage capacity limitation
        self.assertEqual(100, product_dict[SKU1_ID])

        # 4 sku1 cost 4*2 source material (sku3)
        self.assertEqual(100 - 4 * 2, product_dict[SKU3_ID])

        ############################### TICK: 1 ######################################

        # then fix the product rate to 20 every tick, but the manufacture will do nothing, as we have to enough space

        is_done = False

        while not is_done:
            _, _, is_done = env.step({sku1_manufacture_id: ManufactureAction(sku1_storage_index, 20)})

        storage_states = storage_nodes[env.frame_index:sku1_storage_index:storage_features].flatten().astype(np.int)
        manufacture_states = manufacture_nodes[
                             env.frame_index:sku1_data_model_index:manufacture_features].flatten().astype(np.int)

        # but manufacture number is 0
        self.assertEqual(0, manufacture_states[2])

        # but the production rate is same as action
        self.assertEqual(20, manufacture_states[3])

        # so storage remaining space should be 200 - ((96 + 4) + (100 - 4*2))
        self.assertEqual(200 - ((96 + 4) + (100 - 4 * 2)), storage_states[3])

        product_dict = get_product_dict_from_storage(env, env.frame_index, sku1_storage_index)

        # number of sku1 should 100, just reach the avg storage capacity limitation
        self.assertEqual(100, product_dict[SKU1_ID])

        # 4 sku1 cost 4*2 source material (sku3)
        self.assertEqual(100 - 4 * 2, product_dict[SKU3_ID])

    """
    Storage test:

    . take available
        . enough
        . not enough
    . try add products
        . meet whole storage capacity limitation
            . fail if all
            . not fail if all
        . meet avg storage limitation
            .fail if all
            . not fail if all
        . enough space
    . try take products
        . have enough
        . not enough
    . get product number

    """

    def test_storage_take_available(self):
        env = build_env("case_01", 100)

        env.step(None)

        storage_nodes = env.snapshot_list["storage"]
        storage_features = ("id", "capacity", "remaining_space")

        # find first storage unit id
        storage_unit_id = storage_nodes[env.frame_index:0:"id"].flatten().astype(np.int)[0]

        # get the unit reference from env internal
        storage_unit: StorageUnit = env._business_engine.world.get_entity(storage_unit_id)

        storage_states = storage_nodes[env.frame_index:0:storage_features].flatten().astype(np.int)

        capacity = storage_states[1]
        init_remaining_space = storage_states[2]

        init_product_dict = get_product_dict_from_storage(env, env.frame_index, 0)

        # call take_available for each product in storage.
        products_taken = {}
        for product_id, product_number in init_product_dict.items():
            num = np.random.randint(0, product_number)
            actual_num = storage_unit.take_available(product_id, num)

            # we should get the number we want.
            self.assertEqual(num, actual_num)

            products_taken[product_id] = num

        # check if internal state correct
        for product_id, num in products_taken.items():
            remaining_num = storage_unit.product_number[storage_unit.product_index_mapping[product_id]]

            self.assertEqual(init_product_dict[product_id] - num, remaining_num)

        # call env.step will cause states write into snapshot
        env.step(None)

        product_dict = get_product_dict_from_storage(env, env.frame_index, 0)

        for product_id, num in products_taken.items():
            remaining_num = product_dict[product_id]

            self.assertEqual(init_product_dict[product_id] - num, remaining_num)

        # then take more than exist number for 1st product(sku)
        lot_taken_product_id, lot_taken_product_number = product_dict.popitem()

        lot_taken_product_number += 100

        actual_num = storage_unit.take_available(lot_taken_product_id, lot_taken_product_number)

        # we should get all available
        self.assertEqual(actual_num, lot_taken_product_number - 100)

        # take snapshot
        env.step(None)

        product_dict = get_product_dict_from_storage(env, env.frame_index, 0)

        # the product number should be 0, as we took all available
        self.assertEqual(0, product_dict[lot_taken_product_id])


if __name__ == '__main__':
    unittest.main()
