# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest
import numpy as np

from maro.simulator.scenarios.supply_chain import FacilityBase, ManufactureAction
from maro.simulator.scenarios.supply_chain.business_engine import SupplyChainBusinessEngine

from tests.supply_chain.common import build_env, get_product_dict_from_storage, SKU1_ID, SKU2_ID, SKU3_ID, SKU4_ID


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

    def test_manufacture_meet_storage_limitation(self) -> None:
        """Test sku3 manufacturing. -- Supplier_SKU3"""
        env = build_env("case_01", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        supplier_3: FacilityBase = be.world._get_facility_by_name("Supplier_SKU3")
        sku3_storage_index = supplier_3.storage.data_model_index
        manufacture_sku3_unit = supplier_3.products[SKU3_ID].manufacture

        storage_nodes = env.snapshot_list["storage"]

        manufacture_nodes = env.snapshot_list["manufacture"]
        manufacture_features = (
            "id", "facility_id", "start_manufacture_quantity", "sku_id",
        )
        IDX_ID, IDX_FACILITY_ID, IDX_START_MANUFACTURE_QUANTITY, IDX_SKU_ID = 0, 1, 2, 3

        # ############################### TICK: 0 ######################################

        # tick 0 passed, no product manufacturing.
        env.step(None)

        capacities = storage_nodes[env.frame_index:sku3_storage_index:"capacity"].flatten().astype(np.int)
        remaining_spaces = storage_nodes[env.frame_index:sku3_storage_index:"remaining_space"].flatten().astype(np.int)

        # there should be 80 units been taken at the beginning according to the config file.
        # so remaining space should be 20
        self.assertEqual(20, remaining_spaces.sum())
        # capacity is 100 by config
        self.assertEqual(100, capacities.sum())

        product_dict = get_product_dict_from_storage(env, env.frame_index, sku3_storage_index)

        # The product quantity should be same as configuration at beginning.
        # 80 sku3
        self.assertEqual(80, product_dict[SKU3_ID])

        # all the id is greater than 0
        self.assertGreater(manufacture_sku3_unit.id, 0)

        # ######################################################################

        # pass an action to start manufacturing for this tick.
        start_tick = env.tick
        expected_tick = start_tick + 1  # leading time = 1
        action = ManufactureAction(manufacture_sku3_unit.id, 1)

        env.step([action])
        env.step([ManufactureAction(manufacture_sku3_unit.id, 0)])

        while env.tick <= expected_tick:
            env.step(None)

        start_frame = env.business_engine.frame_index(start_tick)
        states = manufacture_nodes[
                 start_frame:manufacture_sku3_unit.data_model_index:manufacture_features
                 ].flatten().astype(np.int)

        # Sku3 produce rate is 1 per tick, so start_manufacture_quantity should be 1.
        self.assertEqual(1, states[IDX_START_MANUFACTURE_QUANTITY])

        expected_frame = env.business_engine.frame_index(expected_tick)
        remaining_spaces = storage_nodes[expected_frame:sku3_storage_index:"remaining_space"].flatten().astype(np.int)

        # now remaining space should be 20 - 1 = 19
        self.assertEqual(20 - 1, remaining_spaces.sum())

        product_dict = get_product_dict_from_storage(env, env.frame_index, sku3_storage_index)

        # sku3 quantity should be 80 + 1
        self.assertEqual(80 + 1, product_dict[SKU3_ID])

        # ######################################################################

        # leave the action as None will cause manufacture unit stop manufacturing.
        env.step(None)

        states = manufacture_nodes[
                 env.frame_index:manufacture_sku3_unit.data_model_index:manufacture_features
                 ].flatten().astype(np.int)

        # so start_manufacture_quantity should be 0
        self.assertEqual(0, states[IDX_START_MANUFACTURE_QUANTITY])

        product_dict = get_product_dict_from_storage(env, env.frame_index, sku3_storage_index)

        # sku3 quantity should be same as last tick
        self.assertEqual(80 + 1, product_dict[SKU3_ID])

        # ######################################################################

        # let it generate 20, but actually it can only procedure 19 because the storage will reach the limitation.
        start_tick = env.tick
        expected_tick = start_tick + 1  # leading time = 1
        env.step([ManufactureAction(manufacture_sku3_unit.id, 20)])
        env.step([ManufactureAction(manufacture_sku3_unit.id, 0)])

        while env.tick <= expected_tick:
            env.step(None)

        start_frame = env.business_engine.frame_index(start_tick)
        states = manufacture_nodes[
                 start_frame:manufacture_sku3_unit.data_model_index:manufacture_features
                 ].flatten().astype(np.int)

        # so start_manufacture_number should be 19 instead 20
        self.assertEqual(19, states[IDX_START_MANUFACTURE_QUANTITY])

        expected_frame = env.business_engine.frame_index(expected_tick)
        remaining_spaces = storage_nodes[expected_frame:sku3_storage_index:"remaining_space"].flatten().astype(np.int)

        # now remaining space should be 0
        self.assertEqual(0, remaining_spaces.sum())

        product_dict = get_product_dict_from_storage(env, env.frame_index, sku3_storage_index)

        # sku3 quantity should be 100
        self.assertEqual(80 + 1 + 19, product_dict[SKU3_ID])

    def test_manufacture_meet_source_lack(self) -> None:
        """Test sku4 manufacturing. -- Supplier_SKU4.
            This sku supplier does not have enough source material at the beginning,
            so it cannot produce anything without consumer purchase."""
        env = build_env("case_01", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        supplier_4: FacilityBase = be.world._get_facility_by_name("Supplier_SKU4")
        sku4_storage_index = supplier_4.storage.data_model_index
        manufacture_sku4_unit = supplier_4.products[SKU4_ID].manufacture

        storage_nodes = env.snapshot_list["storage"]

        manufacture_nodes = env.snapshot_list["manufacture"]
        manufacture_features = (
            "id", "facility_id", "start_manufacture_quantity", "sku_id"
        )
        IDX_ID, IDX_FACILITY_ID, IDX_START_MANUFACTURE_QUANTITY, IDX_SKU_ID = 0, 1, 2, 3

        # ############################### TICK: 0 ######################################

        # tick 0 passed, no product manufacturing.
        env.step(None)

        # capacity is same as configured.
        capacities = storage_nodes[env.frame_index:sku4_storage_index:"capacity"].flatten().astype(np.int)
        self.assertEqual(200, capacities.sum())

        # remaining space should be capacity 200 - (sku4 50 + sku2 0)
        remaining_spaces = storage_nodes[env.frame_index:sku4_storage_index:"remaining_space"].flatten().astype(np.int)
        self.assertEqual(200 - (50 + 0 + 50), remaining_spaces.sum())

        # no manufacture number as we have not pass any action
        manufacture_states = manufacture_nodes[
                             env.frame_index:manufacture_sku4_unit.data_model_index:manufacture_features
                             ].flatten().astype(np.int)

        # manufacture_quantity should be 0
        self.assertEqual(0, manufacture_states[IDX_START_MANUFACTURE_QUANTITY])

        # output product id should be same as configured.
        self.assertEqual(4, manufacture_states[IDX_SKU_ID])

        product_dict = get_product_dict_from_storage(env, env.frame_index, sku4_storage_index)

        # 50 sku4 at beginning
        self.assertEqual(50, product_dict[SKU4_ID])

        # 0 sku2
        self.assertEqual(0, product_dict[SKU2_ID])

        # ############################### TICK: 1 - end ######################################

        is_done = False

        while not is_done:
            # push to the end, the storage should not br changed, no matter what production rate we give it.
            _, _, is_done = env.step([ManufactureAction(manufacture_sku4_unit.id, 10)])

        manufacture_states = manufacture_nodes[
                             env.frame_index:manufacture_sku4_unit.data_model_index:manufacture_features
                             ].flatten().astype(np.int)

        # manufacture_quantity should be 0
        self.assertEqual(0, manufacture_states[IDX_START_MANUFACTURE_QUANTITY])

        # output product id should be same as configured.
        self.assertEqual(SKU4_ID, manufacture_states[IDX_SKU_ID])

        product_dict = get_product_dict_from_storage(env, env.frame_index, sku4_storage_index)

        # 50 sku4 at beginning
        self.assertEqual(50, product_dict[SKU4_ID])

        # 0 sku2
        self.assertEqual(0, product_dict[SKU2_ID])

    def test_manufacture_meet_avg_storage_limitation(self) -> None:
        """Test on sku1 -- Supplier_SKU1.
        It is configured with nearly full initial states."""

        env = build_env("case_01", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        supplier_1: FacilityBase = be.world._get_facility_by_name("Supplier_SKU1")
        sku1_storage_index = supplier_1.storage.data_model_index
        manufacture_sku1_unit = supplier_1.products[SKU1_ID].manufacture

        storage_nodes = env.snapshot_list["storage"]

        manufacture_nodes = env.snapshot_list["manufacture"]
        manufacture_features = (
            "id", "facility_id", "start_manufacture_quantity", "sku_id"
        )
        IDX_ID, IDX_FACILITY_ID, IDX_START_MANUFACTURE_QUANTITY, IDX_SKU_ID = 0, 1, 2, 3

        # ############################### TICK: 0 ######################################

        # tick 0 passed, no product manufacturing, verified in above case, pass checking it here.
        env.step(None)

        # ############################### TICK: 1 ######################################

        # ask sku1 manufacture start manufacturing, rate is 10.
        start_tick = env.tick
        expected_tick = start_tick + 1  # manufacture leading time: 1
        env.step([ManufactureAction(manufacture_sku1_unit.id, 10)])

        env.step([ManufactureAction(manufacture_sku1_unit.id, 0)])
        while env.tick <= expected_tick:
            env.step(None)

        start_frame = env.business_engine.frame_index(start_tick)
        manufacture_states = manufacture_nodes[
                             start_frame:manufacture_sku1_unit.data_model_index:manufacture_features
                             ].flatten().astype(np.int)

        # we can produce 4 sku1, as it will meet storage avg limitation per sku. 4 = 200//2 - 96
        self.assertEqual(200 // 2 - 96, manufacture_states[IDX_START_MANUFACTURE_QUANTITY])

        # so storage remaining space should be 200 - ((96 + 4) + (100 - 4 * 2 sku3/sku1))
        expected_frame = env.business_engine.frame_index(expected_tick)
        remaining_spaces = storage_nodes[
                           expected_frame:manufacture_sku1_unit.data_model_index:"remaining_space"
                           ].flatten().astype(np.int)
        self.assertEqual(200 - ((96 + 4) + (100 - 4 * 2)), remaining_spaces.sum())

        product_dict = get_product_dict_from_storage(env, env.frame_index, sku1_storage_index)

        # The product quantity of sku1 should 100, just reach the avg storage capacity limitation
        self.assertEqual(200 // 2, product_dict[SKU1_ID])

        # 4 sku1 cost 4*2 source material (sku3)
        self.assertEqual(100 - 4 * 2, product_dict[SKU3_ID])

        # ############################### TICK: 1 ######################################

        # then fix the product rate to 20 every tick, but the manufacture will do nothing, as we have no enough space

        is_done = False

        env.step([ManufactureAction(manufacture_sku1_unit.id, 20)])
        while not is_done:
            _, _, is_done = env.step(None)

        manufacture_states = manufacture_nodes[
                             env.frame_index:manufacture_sku1_unit.data_model_index:manufacture_features
                             ].flatten().astype(np.int)

        # but manufacture number is 0
        self.assertEqual(0, manufacture_states[IDX_START_MANUFACTURE_QUANTITY])

        # so storage remaining space should be 200 - ((96 + 4) + (100 - 4*2))
        remaining_spaces = storage_nodes[
                           env.frame_index:sku1_storage_index:"remaining_space"
                           ].flatten().astype(np.int)
        self.assertEqual(200 - ((96 + 4) + (100 - 4 * 2)), remaining_spaces.sum())

        product_dict = get_product_dict_from_storage(env, env.frame_index, sku1_storage_index)

        # The product quantity of sku1 should 100, just reach the avg storage capacity limitation
        self.assertEqual(100, product_dict[SKU1_ID])

        # 4 sku1 cost 4*2 source material (sku3)
        self.assertEqual(100 - 4 * 2, product_dict[SKU3_ID])

    def test_simple_manufacture_without_using_source(self) -> None:
        """Test sku2 simple manufacturing. -- Supplier_SKU2"""
        env = build_env("case_01", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        supplier_2: FacilityBase = be.world._get_facility_by_name("Supplier_SKU2")
        sku2_storage_index = supplier_2.storage.data_model_index
        manufacture_sku2_unit = supplier_2.products[SKU2_ID].manufacture

        storage_nodes = env.snapshot_list["storage"]

        manufacture_nodes = env.snapshot_list["manufacture"]
        manufacture_features = (
            "id", "facility_id", "start_manufacture_quantity", "sku_id",
        )
        IDX_ID, IDX_FACILITY_ID, IDX_START_MANUFACTURE_QUANTITY, IDX_SKU_ID = 0, 1, 2, 3

        # ############################### TICK: 0 ######################################

        # tick 0 passed, no product manufacturing.
        env.step(None)

        capacities = storage_nodes[env.frame_index:sku2_storage_index:"capacity"].flatten().astype(np.int)
        remaining_spaces = storage_nodes[env.frame_index:sku2_storage_index:"remaining_space"].flatten().astype(np.int)

        # there should be 50 + 50 units been taken at the beginning according to the config file.
        # so remaining space should be 200 - (50 + 50) = 100
        self.assertEqual(100, remaining_spaces.sum())
        # capacity is 200 by config
        self.assertEqual(200, capacities.sum())

        product_dict = get_product_dict_from_storage(env, env.frame_index, sku2_storage_index)

        # The product quantity should be same as configuration at beginning.
        # 50 sku2, 50 sku1
        self.assertEqual(50, product_dict[SKU2_ID])
        self.assertEqual(50, product_dict[SKU1_ID])

        # all the id is greater than 0
        self.assertGreater(manufacture_sku2_unit.id, 0)

        # ######################################################################

        # pass an action to start manufacturing for this tick.
        start_tick = env.tick
        expected_tick = start_tick + 1
        action = ManufactureAction(manufacture_sku2_unit.id, 1)

        env.step([action])
        env.step([ManufactureAction(manufacture_sku2_unit.id, 0)])

        while env.tick <= expected_tick:
            env.step(None)

        start_frame = env.business_engine.frame_index(start_tick)
        states = manufacture_nodes[
                 start_frame:manufacture_sku2_unit.data_model_index:manufacture_features
                 ].flatten().astype(np.int)

        # Sku2 produce rate is 1 per tick, and the output per lot is 2, so manufacture_quantity should be 2.
        self.assertEqual(1 * 2, states[IDX_START_MANUFACTURE_QUANTITY])

        expected_frame = env.business_engine.frame_index(expected_tick)
        remaining_spaces = storage_nodes[expected_frame:sku2_storage_index:"remaining_space"].flatten().astype(np.int)

        # now remaining space should be 100 - 1 * 2 = 98 (No consumption of SKU1)
        self.assertEqual(100 - 1 * 2, remaining_spaces.sum())

        product_dict = get_product_dict_from_storage(env, env.frame_index, sku2_storage_index)

        # sku2 quantity should be 50 + 1 * 2
        self.assertEqual(50 + 1 * 2, product_dict[SKU2_ID])
        # sku1 quantity keeps the same
        self.assertEqual(50, product_dict[SKU1_ID])

        # ######################################################################

        # leave the action as None will keep the manufacture rate as 0, so at to stop manufacturing.
        env.step(None)

        states = manufacture_nodes[
                 env.frame_index:manufacture_sku2_unit.data_model_index:manufacture_features
                 ].flatten().astype(np.int)

        # so manufacture_quantity should be 0
        self.assertEqual(0, states[IDX_START_MANUFACTURE_QUANTITY])

        product_dict = get_product_dict_from_storage(env, env.frame_index, sku2_storage_index)

        # sku2 quantity should be same as last tick
        self.assertEqual(50 + 1 * 2, product_dict[SKU2_ID])


if __name__ == '__main__':
    unittest.main()
