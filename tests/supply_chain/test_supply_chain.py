import os
import unittest
from typing import Optional

import numpy as np

from maro.simulator import Env
from maro.simulator.scenarios.supply_chain import (
    ConsumerAction, FacilityBase, ManufactureAction, StorageUnit
)
from maro.simulator.scenarios.supply_chain.business_engine import SupplyChainBusinessEngine
from maro.simulator.scenarios.supply_chain.facilities.facility import FacilityInfo
from maro.simulator.scenarios.supply_chain.order import Order
from maro.simulator.scenarios.supply_chain.units.distribution import Vehicle
from maro.simulator.scenarios.supply_chain.units.storage import AddStrategy


def build_env(case_name: str, durations: int):
    case_folder = os.path.join("tests", "data", "supply_chain", case_name)

    env = Env(scenario="supply_chain", topology=case_folder, durations=durations)

    return env


def get_product_dict_from_storage(env: Env, frame_index: int, node_index: int):
    product_list = env.snapshot_list["storage"][frame_index:node_index:"product_list"].flatten().astype(np.int)
    product_quantity = env.snapshot_list["storage"][frame_index:node_index:"product_quantity"].flatten().astype(np.int)

    return {product_id: quantity for product_id, quantity in zip(product_list, product_quantity)}


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

    def test_manufacture_meet_storage_limitation(self) -> None:
        """Test sku3 manufacturing. -- Supplier_SKU3"""
        env = build_env("case_01", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        storage_nodes = env.snapshot_list["storage"]

        manufacture_nodes = env.snapshot_list["manufacture"]
        manufacture_number = len(manufacture_nodes)
        manufacture_features = (
            "id", "facility_id", "manufacture_quantity", "product_id",
        )
        IDX_ID, IDX_FACILITY_ID, IDX_MANUFACTURE_QUANTITY, IDX_PRODUCT_ID = 0, 1, 2, 3

        # ############################### TICK: 0 ######################################

        # tick 0 passed, no product manufacturing.
        env.step(None)

        states = manufacture_nodes[
            env.frame_index::manufacture_features
        ].flatten().reshape(manufacture_number, -1).astype(np.int)

        # try to find which one is sku3 manufacture unit.
        sku3_data_model_index: Optional[int] = None
        sku3_manufacture_id: Optional[int] = None
        sku3_facility_id: Optional[int] = None
        for index, state in enumerate(states):
            # Id of sku3 is 3.
            if state[IDX_PRODUCT_ID] == SKU3_ID:
                sku3_data_model_index = index
                sku3_manufacture_id = state[IDX_ID]
                sku3_facility_id = state[IDX_FACILITY_ID]
        self.assertTrue(all([
            sku3_data_model_index is not None,
            sku3_manufacture_id is not None,
            sku3_facility_id is not None,
        ]))

        # try to find sku3's storage from env.summary
        sku3_facility_info: FacilityInfo = env.summary["node_mapping"]["facilities"][sku3_facility_id]
        sku3_storage_index = sku3_facility_info.storage_info.node_index

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
        self.assertGreater(sku3_manufacture_id, 0)

        # ############################### TICK: 1 ######################################

        # pass an action to start manufacturing for this tick.
        action = ManufactureAction(sku3_manufacture_id, 1)

        env.step([action])

        states = manufacture_nodes[env.frame_index:sku3_data_model_index:manufacture_features].flatten().astype(np.int)

        # Sku3 produce rate is 1 per tick, so manufacture_quantity should be 1.
        self.assertEqual(1, states[IDX_MANUFACTURE_QUANTITY])

        remaining_spaces = storage_nodes[env.frame_index:sku3_storage_index:"remaining_space"].flatten().astype(np.int)

        # now remaining space should be 20 - 1 = 19
        self.assertEqual(20 - 1, remaining_spaces.sum())

        product_dict = get_product_dict_from_storage(env, env.frame_index, sku3_storage_index)

        # sku3 quantity should be 80 + 1
        self.assertEqual(80 + 1, product_dict[SKU3_ID])

        # ############################### TICK: 2 ######################################

        # leave the action as None will cause manufacture unit stop manufacturing.
        env.step(None)

        states = manufacture_nodes[env.frame_index:sku3_data_model_index:manufacture_features].flatten().astype(np.int)

        # so manufacture_quantity should be 0
        self.assertEqual(0, states[IDX_MANUFACTURE_QUANTITY])

        product_dict = get_product_dict_from_storage(env, env.frame_index, sku3_storage_index)

        # sku3 quantity should be same as last tick
        self.assertEqual(80 + 1, product_dict[SKU3_ID])

        # ############################### TICK: 3 ######################################

        # let is generate 20, but actually it can only procedure 19 because the storage will reach the limitation
        env.step([ManufactureAction(sku3_manufacture_id, 20)])

        states = manufacture_nodes[env.frame_index:sku3_data_model_index:manufacture_features].flatten().astype(np.int)

        # so manufacture_number should be 19 instead 20
        self.assertEqual(19, states[IDX_MANUFACTURE_QUANTITY])

        remaining_spaces = storage_nodes[env.frame_index:sku3_storage_index:"remaining_space"].flatten().astype(np.int)

        # now remaining space should be 0
        self.assertEqual(0, remaining_spaces.sum())

        product_dict = get_product_dict_from_storage(env, env.frame_index, sku3_storage_index)

        # sku3 quantity should be 100
        self.assertEqual(80 + 1 + 19, product_dict[SKU3_ID])

    def test_manufacture_meet_source_lack(self) -> None:
        """Test sku4 manufacturing. -- Supplier_SKU4.
        This sku supplier does not have enough source material at the begining,
        so it cannot produce anything without consumer purchase."""
        env = build_env("case_01", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        storage_nodes = env.snapshot_list["storage"]

        manufacture_nodes = env.snapshot_list["manufacture"]
        manufacture_number = len(manufacture_nodes)
        manufacture_features = (
            "id", "facility_id", "manufacture_quantity", "product_id", "unit_product_cost",
        )
        IDX_ID, IDX_FACILITY_ID, IDX_MANUFACTURE_QUANTITY, IDX_PRODUCT_ID, IDX_UNIT_PRODUCT_COST = 0, 1, 2, 3, 4

        # ############################### TICK: 0 ######################################

        # tick 0 passed, no product manufacturing.
        env.step(None)

        states = manufacture_nodes[
            env.frame_index::manufacture_features
        ].flatten().reshape(manufacture_number, -1).astype(np.int)

        # try to find which one is sku3 manufacture unit.
        sku4_data_model_index: Optional[int] = None
        sku4_manufacture_id: Optional[int] = None
        sku4_facility_id: Optional[int] = None
        for index, state in enumerate(states):
            # Id of sku4 is 4.
            if state[IDX_PRODUCT_ID] == SKU4_ID:
                sku4_data_model_index = index
                sku4_manufacture_id = state[IDX_ID]
                sku4_facility_id = state[IDX_FACILITY_ID]
        self.assertTrue(all([
            sku4_data_model_index is not None,
            sku4_manufacture_id is not None,
            sku4_facility_id is not None,
        ]))

        # try to find sku4's storage from env.summary
        sku4_facility_info: FacilityInfo = env.summary["node_mapping"]["facilities"][sku4_facility_id]
        sku4_storage_index = sku4_facility_info.storage_info.node_index

        # capacity is same as configured.
        capacities = storage_nodes[env.frame_index:sku4_storage_index:"capacity"].flatten().astype(np.int)
        self.assertEqual(200, capacities.sum())

        # remaining space should be capacity 200 - (sku4 50 + sku2 0)
        remaining_spaces = storage_nodes[env.frame_index:sku4_storage_index:"remaining_space"].flatten().astype(np.int)
        self.assertEqual(200 - (50 + 0 + 50), remaining_spaces.sum())

        # no manufacture number as we have not pass any action
        manufacture_states = manufacture_nodes[
            env.frame_index:sku4_data_model_index:manufacture_features
        ].flatten().astype(np.int)

        # manufacture_quantity should be 0
        self.assertEqual(0, manufacture_states[IDX_MANUFACTURE_QUANTITY])

        # output product id should be same as configured.
        self.assertEqual(4, manufacture_states[IDX_PRODUCT_ID])

        # product unit cost should be same as configured.
        self.assertEqual(4, manufacture_states[IDX_UNIT_PRODUCT_COST])

        product_dict = get_product_dict_from_storage(env, env.frame_index, sku4_storage_index)

        # 50 sku4 at beginning
        self.assertEqual(50, product_dict[SKU4_ID])

        # 0 sku2
        self.assertEqual(0, product_dict[SKU2_ID])

        # ############################### TICK: 1 - end ######################################

        is_done = False

        while not is_done:
            # push to the end, the storage should not changed, no matter what production rate we give it.
            _, _, is_done = env.step([ManufactureAction(sku4_manufacture_id, 10)])

        manufacture_states = manufacture_nodes[
            env.frame_index:sku4_data_model_index:manufacture_features
        ].flatten().astype(np.int)

        # manufacture_quantity should be 0
        self.assertEqual(0, manufacture_states[IDX_MANUFACTURE_QUANTITY])

        # output product id should be same as configured.
        self.assertEqual(SKU4_ID, manufacture_states[IDX_PRODUCT_ID])

        # product unit cost should be same as configured.
        self.assertEqual(4, manufacture_states[IDX_UNIT_PRODUCT_COST])

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

        storage_nodes = env.snapshot_list["storage"]

        manufacture_nodes = env.snapshot_list["manufacture"]
        manufacture_number = len(manufacture_nodes)
        manufacture_features = (
            "id", "facility_id", "manufacture_quantity", "product_id", "unit_product_cost",
        )
        IDX_ID, IDX_FACILITY_ID, IDX_MANUFACTURE_QUANTITY, IDX_PRODUCT_ID, IDX_UNIT_PRODUCT_COST = 0, 1, 2, 3, 4

        # ############################### TICK: 0 ######################################

        # tick 0 passed, no product manufacturing, verified in above case, pass checking it here.
        env.step(None)

        states = manufacture_nodes[
            env.frame_index::manufacture_features
        ].flatten().reshape(manufacture_number, -1).astype(np.int)
        # try to find which one is sku3 manufacture unit.
        sku1_data_model_index: Optional[int] = None
        sku1_manufacture_id: Optional[int] = None
        sku1_facility_id: Optional[int] = None
        for index, state in enumerate(states):
            # Id of sku1 is 1.
            if state[IDX_PRODUCT_ID] == SKU1_ID:
                sku1_data_model_index = index
                sku1_manufacture_id = state[IDX_ID]
                sku1_facility_id = state[IDX_FACILITY_ID]
        self.assertTrue(all([
            sku1_data_model_index is not None,
            sku1_manufacture_id is not None,
            sku1_facility_id is not None,
        ]))

        sku1_facility_info: FacilityInfo = env.summary["node_mapping"]["facilities"][sku1_facility_id]
        sku1_storage_index = sku1_facility_info.storage_info.node_index

        # ############################### TICK: 1 ######################################

        # ask sku1 manufacture start manufacturing, rate is 10.
        env.step([ManufactureAction(sku1_manufacture_id, 10)])

        manufacture_states = manufacture_nodes[
            env.frame_index:sku1_data_model_index:manufacture_features
        ].flatten().astype(np.int)

        # we can produce 4 sku1, as it will meet storage avg limitation per sku. 4 = 200//2 - 96
        self.assertEqual(200 // 2 - 96, manufacture_states[IDX_MANUFACTURE_QUANTITY])

        # so storage remaining space should be 200 - ((96 + 4) + (100 - 4 * 2 sku3/sku1))
        remaining_spaces = storage_nodes[
            env.frame_index:sku1_data_model_index:"remaining_space"
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

        while not is_done:
            _, _, is_done = env.step([ManufactureAction(sku1_storage_index, 20)])

        manufacture_states = manufacture_nodes[
            env.frame_index:sku1_data_model_index:manufacture_features
        ].flatten().astype(np.int)

        # but manufacture number is 0
        self.assertEqual(0, manufacture_states[IDX_MANUFACTURE_QUANTITY])

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

        storage_nodes = env.snapshot_list["storage"]

        manufacture_nodes = env.snapshot_list["manufacture"]
        manufacture_number = len(manufacture_nodes)
        manufacture_features = (
            "id", "facility_id", "manufacture_quantity", "product_id",
        )
        IDX_ID, IDX_FACILITY_ID, IDX_MANUFACTURE_QUANTITY, IDX_PRODUCT_ID = 0, 1, 2, 3

        # ############################### TICK: 0 ######################################

        # tick 0 passed, no product manufacturing.
        env.step(None)

        states = manufacture_nodes[
            env.frame_index::manufacture_features
        ].flatten().reshape(manufacture_number, -1).astype(np.int)

        # try to find which one is sku2 manufacture unit.
        sku2_data_model_index: Optional[int] = None
        sku2_manufacture_id: Optional[int] = None
        sku2_facility_id: Optional[int] = None
        for index, state in enumerate(states):
            if state[IDX_PRODUCT_ID] == SKU2_ID:
                sku2_data_model_index = index
                sku2_manufacture_id = state[IDX_ID]
                sku2_facility_id = state[IDX_FACILITY_ID]
        self.assertTrue(all([
            sku2_data_model_index is not None,
            sku2_manufacture_id is not None,
            sku2_facility_id is not None,
        ]))

        # try to find sku2's storage from env.summary
        sku2_facility_info: FacilityInfo = env.summary["node_mapping"]["facilities"][sku2_facility_id]
        sku2_storage_index = sku2_facility_info.storage_info.node_index

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
        self.assertGreater(sku2_manufacture_id, 0)

        # ############################### TICK: 1 ######################################

        # pass an action to start manufacturing for this tick.
        action = ManufactureAction(sku2_manufacture_id, 1)

        env.step([action])

        states = manufacture_nodes[env.frame_index:sku2_data_model_index:manufacture_features].flatten().astype(np.int)

        # Sku2 produce rate is 1 per tick, so manufacture_quantity should be 1.
        self.assertEqual(1, states[IDX_MANUFACTURE_QUANTITY])

        remaining_spaces = storage_nodes[env.frame_index:sku2_storage_index:"remaining_space"].flatten().astype(np.int)

        # now remaining space should be 100 - 1 = 99 (No consumption of SKU1)
        self.assertEqual(100 - 1, remaining_spaces.sum())

        product_dict = get_product_dict_from_storage(env, env.frame_index, sku2_storage_index)

        # sku2 quantity should be 80 + 1
        self.assertEqual(50 + 1, product_dict[SKU2_ID])
        # sku1 quantity keeps the same
        self.assertEqual(50, product_dict[SKU1_ID])

        # ############################### TICK: 2 ######################################

        # leave the action as None will cause manufacture unit stop manufacturing.
        env.step(None)

        states = manufacture_nodes[env.frame_index:sku2_data_model_index:manufacture_features].flatten().astype(np.int)

        # so manufacture_quantity should be 0
        self.assertEqual(0, states[IDX_MANUFACTURE_QUANTITY])

        product_dict = get_product_dict_from_storage(env, env.frame_index, sku2_storage_index)

        # sku2 quantity should be same as last tick
        self.assertEqual(50 + 1, product_dict[SKU2_ID])

    """
    Storage test:

    . take available
        . enough
        . not enough
    . try add products
        . meet whole storage capacity limitation
            . fail if all
            . not fail if all
        . enough space
    . try take products
        . have enough
        . not enough
    . get product quantity

    """

    def test_storage_get_product_quantity_and_capacity_and_remaining_space(self) -> None:
        """Supplier_SKU1"""
        env = build_env("case_01", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        env.step(None)

        supplier_1: FacilityBase = be.world._get_facility_by_name("Supplier_SKU1")
        storage_unit: StorageUnit = supplier_1.storage
        storage_node_index = storage_unit.data_model_index

        storage_nodes = env.snapshot_list["storage"]

        # ######################### Product Quantity ###########################
        init_product_dict = get_product_dict_from_storage(env, env.frame_index, storage_node_index)
        self.assertEqual(2, len(init_product_dict))

        # Inside StorageUnit
        self.assertEqual(96, storage_unit._product_level[SKU1_ID])
        self.assertEqual(100, storage_unit._product_level[SKU3_ID])
        # In Snapshot
        self.assertEqual(96, init_product_dict[SKU1_ID])
        self.assertEqual(100, init_product_dict[SKU3_ID])

        # ######################### Capacity ###########################
        capacities = storage_nodes[env.frame_index:storage_node_index:"capacity"].flatten().astype(np.int)
        self.assertEqual(200, storage_unit.capacity)
        self.assertEqual(200, capacities.sum())

        # ######################### Remaining Space ###########################
        init_remaining_spaces = storage_nodes[
            env.frame_index:storage_node_index:"remaining_space"
        ].flatten().astype(np.int)
        self.assertEqual(200 - 96 - 100, storage_unit.remaining_space)
        self.assertEqual(200 - 96 - 100, init_remaining_spaces.sum())

        # ######################### Remaining Space ###########################
        # Should not change even after reset
        env.reset()
        env.step(None)

        # ######################### Product Quantity ###########################
        init_product_dict = get_product_dict_from_storage(env, env.frame_index, storage_node_index)

        # Inside StorageUnit
        self.assertEqual(96, storage_unit._product_level[SKU1_ID])
        self.assertEqual(100, storage_unit._product_level[SKU3_ID])
        # In Snapshot
        self.assertEqual(96, init_product_dict[SKU1_ID])
        self.assertEqual(100, init_product_dict[SKU3_ID])

        # ######################### Capacity ###########################
        capacities = storage_nodes[env.frame_index:storage_node_index:"capacity"].flatten().astype(np.int)
        self.assertEqual(200, storage_unit.capacity)
        self.assertEqual(200, capacities.sum())

        # ######################### Remaining Space ###########################
        init_remaining_spaces = storage_nodes[
            env.frame_index:storage_node_index:"remaining_space"
        ].flatten().astype(np.int)
        self.assertEqual(200 - 96 - 100, storage_unit.remaining_space)
        self.assertEqual(200 - 96 - 100, init_remaining_spaces.sum())

    def test_storage_take_available(self) -> None:
        """Facility with single SKU. -- Supplier_SKU3"""
        env = build_env("case_01", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        env.step(None)

        supplier_3: FacilityBase = be.world._get_facility_by_name("Supplier_SKU3")
        storage_unit: StorageUnit = supplier_3.storage
        storage_node_index = storage_unit.data_model_index

        # ######################### Take a right amount of quantity ##############################
        # Call take_available to take 40 sku3 in storage.
        actual_quantity = storage_unit.take_available(SKU3_ID, 40)
        self.assertEqual(40, actual_quantity)

        # Check if remaining quantity correct
        self.assertEqual(80 - 40, storage_unit._product_level[SKU3_ID])

        # call env.step will cause states write into snapshot
        env.step(None)

        # Check if the snapshot status correct
        product_dict = get_product_dict_from_storage(env, env.frame_index, storage_node_index)
        self.assertEqual(80 - 40, product_dict[SKU3_ID])

        # ######################### Take more than existing ##############################
        try_taken_quantity = (80 - 40) + 10
        actual_quantity = storage_unit.take_available(SKU3_ID, try_taken_quantity)
        # We should get all available
        self.assertEqual(actual_quantity, try_taken_quantity - 10)

        # take snapshot
        env.step(None)

        product_dict = get_product_dict_from_storage(env, env.frame_index, storage_node_index)

        # The product quantity should be 0, as we took all available
        self.assertEqual(0, product_dict[SKU3_ID])

    def test_storage_try_add_products(self) -> None:
        """Facility with multiple SKUs -- Supplier_SKU2
        NOTE:
            try_add_products method do not check avg storage capacity checking, so we will ignore it here.

        """
        env = build_env("case_01", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        env.step(None)

        supplier_2 = be.world._get_facility_by_name("Supplier_SKU2")
        storage_unit = supplier_2.storage
        storage_node_index = storage_unit.data_model_index

        storage_nodes = env.snapshot_list["storage"]

        capacities = storage_nodes[env.frame_index:storage_node_index:"capacity"].flatten().astype(np.int)

        init_remaining_spaces = storage_nodes[
            env.frame_index:storage_node_index:"remaining_space"
        ].flatten().astype(np.int)

        init_product_dict = get_product_dict_from_storage(env, env.frame_index, storage_node_index)

        # ############################### IgnoreUpperBound AllOrNothing ######################################
        # 100 // 2 = 50
        avg_max_product_quantity = init_remaining_spaces.sum() // len(init_product_dict)
        self.assertEqual(50, avg_max_product_quantity)

        products_to_put = {
            SKU1_ID: 50 + 1,
            SKU2_ID: 50 + 1,
        }

        result = storage_unit.try_add_products(
            products_to_put,
            add_strategy=AddStrategy.IgnoreUpperBoundAllOrNothing,
        )
        # the method will return an empty dictionary if fail to add
        self.assertEqual(0, len(result))

        # so remaining space should not change
        self.assertEqual(100, storage_unit.remaining_space)

        # each product quantity should be same as before
        self.assertEqual(50, storage_unit._product_level[SKU2_ID])
        self.assertEqual(50, storage_unit._product_level[SKU1_ID])

        # ############################### IgnoreUpperBound AddInOrder ######################################
        # Part of the product will be added to storage, and cause remaining space being 0
        storage_unit.try_add_products(
            products_to_put,
            add_strategy=AddStrategy.IgnoreUpperBoundAddInOrder,
        )
        # all sku1 would be added successfully
        self.assertEqual(50 + (50 + 1), storage_unit._product_level[SKU1_ID])
        self.assertEqual(50 + (100 - (50 + 1)), storage_unit._product_level[SKU2_ID])

        self.assertEqual(0, storage_unit.remaining_space)

        # take snapshot
        env.step(None)

        # remaining space in snapshot should be 0
        remaining_spaces = storage_nodes[env.frame_index:storage_node_index:"remaining_space"].flatten().astype(np.int)
        self.assertEqual(0, remaining_spaces.sum())

        product_dict = get_product_dict_from_storage(env, env.frame_index, storage_node_index)
        self.assertEqual(50 + 51, product_dict[SKU1_ID])
        self.assertEqual(50 + 49, product_dict[SKU2_ID])

        # total product quantity should be same as capacity
        self.assertEqual(capacities.sum(), sum(product_dict.values()))

        # ######################################################################
        # reset the env for next case
        env.reset()

        # check the state after reset
        self.assertEqual(capacities.sum(), storage_unit.capacity)
        self.assertEqual(init_remaining_spaces.sum(), storage_unit.remaining_space)

        for product_id, product_quantity in init_product_dict.items():
            self.assertEqual(product_quantity, storage_unit._product_level[product_id])

        # ############################### IgnoreUpperBound Proportional ######################################
        products_to_put = {
            SKU1_ID: 50,
            SKU2_ID: 150,
        }
        # Part of the product will be added to storage, and cause remaining space being 0
        storage_unit.try_add_products(
            products_to_put,
            add_strategy=AddStrategy.IgnoreUpperBoundProportional,
        )
        # Only 100 // (50 + 150) = 1/2 incoming products can be added.
        self.assertEqual(50 + 50 // 2, storage_unit._product_level[SKU1_ID])
        self.assertEqual(50 + 150 // 2, storage_unit._product_level[SKU2_ID])

        self.assertEqual(0, storage_unit.remaining_space)

        # take snapshot
        env.step(None)

        # remaining space in snapshot should be 0
        remaining_spaces = storage_nodes[env.frame_index:storage_node_index:"remaining_space"].flatten().astype(np.int)
        self.assertEqual(0, remaining_spaces.sum())

        product_dict = get_product_dict_from_storage(env, env.frame_index, storage_node_index)
        self.assertEqual(50 + 25, product_dict[SKU1_ID])
        self.assertEqual(50 + 75, product_dict[SKU2_ID])

        # ######################################################################
        # reset the env for next case
        env.reset()

        # check the state after reset
        self.assertEqual(capacities.sum(), storage_unit.capacity)
        self.assertEqual(init_remaining_spaces.sum(), storage_unit.remaining_space)

        for product_id, product_quantity in init_product_dict.items():
            self.assertEqual(product_quantity, storage_unit._product_level[product_id])

        # ############################### LimitedByUpperBound ######################################
        products_to_put = {
            SKU1_ID: 60,
            SKU2_ID: 40,
        }

        storage_unit.try_add_products(
            products_to_put,
            add_strategy=AddStrategy.LimitedByUpperBound,
        )
        # the default upper bound is the avg capacity, so it would be 100 for both sku1 and sku2
        self.assertEqual(50 + min(100 - 50, 60), storage_unit._product_level[SKU1_ID])
        self.assertEqual(50 + min(100 - 50, 40), storage_unit._product_level[SKU2_ID])

        # 10 = 200 - (50 + 50) - (50 + 40)
        self.assertEqual(10, storage_unit.remaining_space)

        # take snapshot
        env.step(None)

        remaining_spaces = storage_nodes[env.frame_index:storage_node_index:"remaining_space"].flatten().astype(np.int)
        self.assertEqual(10, remaining_spaces.sum())

        product_dict = get_product_dict_from_storage(env, env.frame_index, storage_node_index)
        self.assertEqual(50 + 50, product_dict[SKU1_ID])
        self.assertEqual(50 + 40, product_dict[SKU2_ID])

    def test_storage_try_take_products(self) -> None:
        """Facility with single SKU. -- Supplier_SKU3"""
        env = build_env("case_01", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        env.step(None)

        supplier_3: FacilityBase = be.world._get_facility_by_name("Supplier_SKU3")
        storage_unit: StorageUnit = supplier_3.storage
        storage_node_index = storage_unit.data_model_index

        storage_nodes = env.snapshot_list["storage"]

        # ############################### Take more than existing ######################################
        product_to_take = {
            SKU3_ID: 81,
        }
        # which this setting, it will return false, as no enough product for ous
        self.assertFalse(storage_unit.try_take_products(product_to_take))

        # so remaining space and product quantity should same as before
        self.assertEqual(100 - 80, storage_unit.remaining_space)
        self.assertEqual(80, storage_unit._product_level[SKU3_ID])

        # ############################### Take all ######################################
        # try to get all products
        product_to_take = {
            SKU3_ID: 80,
        }
        self.assertTrue(storage_unit.try_take_products(product_to_take))

        # now the remaining space should be same as capacity as we take all
        self.assertEqual(100, storage_unit.remaining_space)

        # take snapshot
        env.step(None)

        capacities = storage_nodes[env.frame_index:storage_node_index:"capacity"].flatten().astype(np.int)
        remaining_spaces = storage_nodes[env.frame_index:storage_node_index:"remaining_space"].flatten().astype(np.int)

        # remaining space should be same as capacity in snapshot
        self.assertEqual(capacities.sum(), remaining_spaces.sum())

    def test_storage_upper_bound(self) -> None:
        """Warehouse_001."""
        env = build_env("case_02", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        env.step(None)

        warehouse: FacilityBase = be.world._get_facility_by_name("Warehouse_001")
        storage_unit: StorageUnit = warehouse.storage
        storage_node_index = storage_unit.data_model_index

        storage_nodes = env.snapshot_list["storage"]

        # ######################### Check the storage upper bound for each sku ##############################
        capacities = storage_nodes[env.frame_index:storage_node_index:"capacity"].flatten().astype(np.int)
        remaining_spaces = storage_nodes[env.frame_index:storage_node_index:"remaining_space"].flatten().astype(np.int)

        # The capacity should be same as the config
        self.assertEqual(100, capacities[0])
        self.assertEqual(100, capacities[1])
        self.assertEqual(100 + 100, capacities.sum())

        # All SKUs are saved in sub_storage 1
        self.assertEqual(100 - 10 - 10 - 10, remaining_spaces[0])
        self.assertEqual(100, remaining_spaces[1])
        self.assertEqual(100 - 10 - 10 - 10 + 100, remaining_spaces.sum())

        self.assertEqual(40 - 10, storage_unit.get_product_max_remaining_space(SKU1_ID))
        self.assertEqual((100 - 40) // 2 - 10, storage_unit.get_product_max_remaining_space(SKU2_ID))
        self.assertEqual((100 - 40) // 2 - 10, storage_unit.get_product_max_remaining_space(SKU3_ID))

        # ######################### Test the try add with Limited by pre-set upper bound ##############################
        result = storage_unit.try_add_products({SKU1_ID: 50}, add_strategy=AddStrategy.LimitedByUpperBound)
        self.assertEqual(40 - 10, result[SKU1_ID])
        self.assertEqual(0, storage_unit.get_product_max_remaining_space(SKU1_ID))

        env.step(None)
        remaining_spaces = storage_nodes[env.frame_index:storage_node_index:"remaining_space"].flatten().astype(np.int)
        self.assertEqual(100 - 40 - 10 - 10, remaining_spaces[0])

    """
    Consumer test:

    . initial state
    . state after reset
    . set_action directly from code
    . set_action by env.step
    . call on_order_reception directly to simulation order arrived
    . call update_open_orders directly

    """

    def test_consumer_init_state(self) -> None:
        """Consumer of sku3 in Supplier_SKU1."""
        env = build_env("case_01", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        supplier_1: FacilityBase = be.world._get_facility_by_name("Supplier_SKU1")
        sku3_consumer_unit = supplier_1.products[SKU3_ID].consumer

        consumer_node_index = sku3_consumer_unit.data_model_index

        features = ("id", "facility_id", "product_id", "order_base_cost", "purchased", "received", "order_product_cost")
        IDX_ID, IDX_FACILITY_ID, IDX_PRODUCT_ID, IDX_ORDER_COST = 0, 1, 2, 3
        IDX_PURCHASED, IDX_RECEIVED, IDX_ORDER_PRODUCT_COST = 4, 5, 6

        consumer_nodes = env.snapshot_list["consumer"]

        # check initial state
        self.assertEqual(0, sku3_consumer_unit._received)
        self.assertEqual(0, sku3_consumer_unit._purchased)
        self.assertEqual(0, sku3_consumer_unit._order_product_cost)

        # check data model state
        # order cost from configuration
        self.assertEqual(200, sku3_consumer_unit._unit_order_cost)

        # NOTE: 0 is an invalid(initial) id
        self.assertEqual(0, sku3_consumer_unit.data_model.purchased)
        self.assertEqual(0, sku3_consumer_unit.data_model.received)
        self.assertEqual(0, sku3_consumer_unit.data_model.order_product_cost)

        # check sources
        for source_facility_id in sku3_consumer_unit.source_facility_id_list:
            source_facility: FacilityBase = be.world.get_facility_by_id(source_facility_id)

            # check if source facility contains the sku3 config
            self.assertTrue(SKU3_ID in source_facility.skus)

        env.step(None)

        # check state
        states = consumer_nodes[env.frame_index:consumer_node_index:features].flatten().astype(np.int)

        self.assertEqual(sku3_consumer_unit.id, states[IDX_ID])
        self.assertEqual(sku3_consumer_unit.facility.id, states[IDX_FACILITY_ID])
        self.assertEqual(SKU3_ID, states[IDX_PRODUCT_ID])
        self.assertEqual(0, states[IDX_ORDER_COST])

        env.reset()
        env.step(None)

        states = consumer_nodes[env.frame_index:consumer_node_index:features].flatten().astype(np.int)

        # Nothing happened at tick 0, so most states will be 0
        self.assertEqual(0, states[IDX_PURCHASED])
        self.assertEqual(0, states[IDX_RECEIVED])
        self.assertEqual(0, states[IDX_ORDER_PRODUCT_COST])

        self.assertEqual(sku3_consumer_unit.id, states[IDX_ID])
        self.assertEqual(SKU3_ID, states[IDX_PRODUCT_ID])

    def test_consumer_action(self) -> None:
        """Consumer of sku3 in Supplier_SKU1, which would purchase from Supplier_SKU3."""
        env = build_env("case_01", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        env.step(None)

        supplier_1: FacilityBase = be.world._get_facility_by_name("Supplier_SKU1")
        supplier_3: FacilityBase = be.world._get_facility_by_name("Supplier_SKU3")
        sku3_consumer_unit = supplier_1.products[SKU3_ID].consumer

        consumer_node_index = sku3_consumer_unit.data_model_index

        features = ("id", "facility_id", "product_id", "order_base_cost", "purchased", "received", "order_product_cost")
        IDX_ID, IDX_FACILITY_ID, IDX_PRODUCT_ID, IDX_ORDER_COST = 0, 1, 2, 3
        IDX_PURCHASED, IDX_RECEIVED, IDX_ORDER_PRODUCT_COST = 4, 5, 6

        consumer_nodes = env.snapshot_list["consumer"]

        # ############################### Test Action with 0 quantity ######################################
        # zero quantity will be ignore
        action_with_zero = ConsumerAction(sku3_consumer_unit.id, SKU3_ID, supplier_3.id, 0, "train")
        env.step([action_with_zero])

        states = consumer_nodes[env.frame_index:consumer_node_index:features].flatten().astype(np.int)

        # Nothing happened at tick 0, at the action will be recorded
        self.assertEqual(action_with_zero.product_id, states[IDX_PRODUCT_ID])
        self.assertEqual(action_with_zero.quantity, states[IDX_PURCHASED])

        self.assertEqual(sku3_consumer_unit.id, states[IDX_ID])
        self.assertEqual(SKU3_ID, states[IDX_PRODUCT_ID])

        # ############################### Test Action with positive quantity ######################################
        action = ConsumerAction(sku3_consumer_unit.id, SKU3_ID, supplier_3.id, 1, "train")
        env.step([action])

        self.assertEqual(action.quantity, sku3_consumer_unit._purchased)
        self.assertEqual(0, sku3_consumer_unit._received)

        states = consumer_nodes[env.frame_index:consumer_node_index:features].flatten().astype(np.int)

        # action field should be recorded
        self.assertEqual(action.product_id, states[IDX_PRODUCT_ID])

        self.assertEqual(action.quantity, states[IDX_PURCHASED])

        # no receives
        self.assertEqual(0, states[IDX_RECEIVED])

    def test_consumer_on_order_reception(self) -> None:
        env = build_env("case_01", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        env.step(None)

        supplier_1: FacilityBase = be.world._get_facility_by_name("Supplier_SKU1")
        supplier_3: FacilityBase = be.world._get_facility_by_name("Supplier_SKU3")
        sku3_consumer_unit = supplier_1.products[SKU3_ID].consumer

        required_quantity = 1
        action = ConsumerAction(sku3_consumer_unit.id, SKU3_ID, supplier_3.id, required_quantity, "train")

        env.step([action])

        # simulate purchased product is arrived by vehicle unit
        sku3_consumer_unit.on_order_reception(supplier_3.id, SKU3_ID, required_quantity, required_quantity)

        # now all order is done
        self.assertEqual(0, sku3_consumer_unit._open_orders[supplier_3.id][SKU3_ID])
        self.assertEqual(required_quantity, sku3_consumer_unit._received)

        # NOTE: we cannot test the received state by calling on_order_reception directly,
        # as it will be cleared by env.step, do it on vehicle unit test.

        env.step(None)

    """
    Vehicle unit test:

    . initial state
    . if vehicle arrive at destination within special vlt
    . schedule job
    . try_load until patient <= 0 to cancel the schedule
    . try_load until patient > 0 to load order
    . try_unload
        . target storage cannot take all
        . target storage can take all
    """

    # def test_vehicle_unit_state(self) -> None:
    #     """Test the first Vehicle of Supplier_SKU3."""
    #     env = build_env("case_02", 100)
    #     be = env.business_engine
    #     assert isinstance(be, SupplyChainBusinessEngine)

    #     supplier_3: FacilityBase = be.world._get_facility_by_name("Supplier_SKU3")
    #     vehicle_unit = supplier_3.distribution.children[0]
    #     self.assertTrue(isinstance(vehicle_unit, VehicleUnit))

    #     # ############################### Check the initial state ######################################
    #     env.step(None)

    #     # Check initial state according to configuration file
    #     self.assertEqual(3, vehicle_unit._max_patient)

    #     # no request
    #     self.assertEqual(0, vehicle_unit.requested_quantity)
    #     # no destination
    #     self.assertIsNone(vehicle_unit._destination)
    #     # no product
    #     self.assertEqual(0, vehicle_unit.product_id)
    #     # no steps
    #     self.assertEqual(0, vehicle_unit._remaining_steps)
    #     # no payloads
    #     self.assertEqual(0, vehicle_unit.payload)
    #     # no steps
    #     self.assertEqual(0, vehicle_unit._steps)

    #     # state in frame
    #     self.assertEqual(0, vehicle_unit.data_model.payload)
    #     self.assertEqual(12, vehicle_unit.data_model.unit_transport_cost)

    #     # ############################### Check the state after reset ######################################
    #     # reset to check again
    #     env.reset()

    #     # check initial state according to configuration file
    #     self.assertEqual(3, vehicle_unit._max_patient)

    #     # no request
    #     self.assertEqual(0, vehicle_unit.requested_quantity)
    #     # no destination
    #     self.assertIsNone(vehicle_unit._destination)
    #     # no product
    #     self.assertEqual(0, vehicle_unit.product_id)
    #     # no steps
    #     self.assertEqual(0, vehicle_unit._remaining_steps)
    #     # no payloads
    #     self.assertEqual(0, vehicle_unit.payload)
    #     # no steps
    #     self.assertEqual(0, vehicle_unit._steps)

    #     # state in frame
    #     self.assertEqual(0, vehicle_unit.data_model.payload)
    #     self.assertEqual(12, vehicle_unit.data_model.unit_transport_cost)

    # def test_vehicle_unit_schedule(self) -> None:
    #     """Schedule the first Vehicle of Supplier_SKU3 to Warehouse_001."""
    #     env = build_env("case_02", 100)
    #     be = env.business_engine
    #     assert isinstance(be, SupplyChainBusinessEngine)

    #     supplier_3 = be.world._get_facility_by_name("Supplier_SKU3")
    #     warehouse_1 = be.world._get_facility_by_name("Warehouse_001")

    #     vehicle_unit: VehicleUnit = supplier_3.distribution.children[0]
    #     self.assertTrue(isinstance(vehicle_unit, VehicleUnit))

    #     # make sure the upstream in the only one supplier in config
    #     self.assertEqual(1, len(warehouse_1.upstream_vlt_infos))
    #     self.assertEqual(1, len(warehouse_1.upstream_vlt_infos[SKU3_ID]))

    #     vehicle_nodes = env.snapshot_list["vehicle"]

    #     features = ("id", "facility_id", "payload", "unit_transport_cost")
    #     IDX_ID, IDX_FACILITY_ID, IDX_PAYLOAD, IDX_UNIT_TRANSPORT_COST = 0, 1, 2, 3

    #     # ############################### Schedule Manually ######################################
    #     self.assertEqual(12, vehicle_unit.data_model.unit_transport_cost)

    #     # step to take snapshot
    #     env.step(None)

    #     # schedule vehicle unit manually, from supplier to warehouse
    #     vlt = 7
    #     vehicle_unit.schedule(warehouse_1, SKU3_ID, 20, vlt)
    #     schedule_tick: int = env.tick
    #     expected_arrival_tick: int = schedule_tick + vlt

    #     # check internal states
    #     self.assertEqual(warehouse_1, vehicle_unit._destination)
    #     self.assertEqual(SKU3_ID, vehicle_unit.product_id)
    #     self.assertEqual(20, vehicle_unit.requested_quantity)
    #     self.assertEqual(vlt, vehicle_unit._remaining_steps)

    #     # Step to flush to data model
    #     env.step(None)

    #     states = vehicle_nodes[env.frame_index:vehicle_unit.data_model_index:features].flatten().astype(np.int)

    #     # source id
    #     self.assertEqual(vehicle_unit.facility.id, states[IDX_FACILITY_ID])
    #     # payload should be 20, as we already env.step
    #     self.assertEqual(20, states[IDX_PAYLOAD])

    #     # ############################### 1 tick before arrival ######################################
    #     while env.tick < expected_arrival_tick - 1:
    #         env.step(None)

    #     states = vehicle_nodes[env.frame_index:vehicle_unit.data_model_index:features].flatten().astype(np.int)

    #     # payload
    #     self.assertEqual(20, states[IDX_PAYLOAD])

    #     # ############################### Arrival tick ######################################
    #     env.step(None)

    #     states = vehicle_nodes[env.frame_index:vehicle_unit.data_model_index:features].flatten().astype(np.int)

    #     # the product is unloaded, vehicle states will be reset to initial
    #     # not destination at first
    #     self.assertIsNone(vehicle_unit._destination)
    #     self.assertEqual(0, vehicle_unit.product_id)
    #     self.assertEqual(0, vehicle_unit._remaining_steps)
    #     self.assertEqual(0, vehicle_unit.payload)
    #     self.assertEqual(0, vehicle_unit._steps)
    #     self.assertEqual(0, vehicle_unit.requested_quantity)

    #     # check states
    #     self.assertEqual(0, states[IDX_PAYLOAD])

    # def test_vehicle_unit_no_patient(self) -> None:
    #     """Test Vehicle no patient by trying to load more products than Supplier_SKU3 has to Warehouse_001."""
    #     env = build_env("case_02", 100)
    #     be = env.business_engine
    #     assert isinstance(be, SupplyChainBusinessEngine)

    #     supplier_3 = be.world._get_facility_by_name("Supplier_SKU3")
    #     warehouse_1 = be.world._get_facility_by_name("Warehouse_001")

    #     vehicle_unit: VehicleUnit = supplier_3.distribution.children[0]
    #     self.assertTrue(isinstance(vehicle_unit, VehicleUnit))

    #     # make sure the upstream in the only one supplier in config
    #     self.assertEqual(1, len(warehouse_1.upstream_vlt_infos))
    #     self.assertEqual(1, len(warehouse_1.upstream_vlt_infos[SKU3_ID]))

    #     vehicle_nodes = env.snapshot_list["vehicle"]

    #     features = ("id", "facility_id", "payload", "unit_transport_cost")
    #     IDX_ID, IDX_FACILITY_ID, IDX_PAYLOAD, IDX_UNIT_TRANSPORT_COST = 0, 1, 2, 3

    #     # ############################### Try to load more than the supplier has ######################################
    #     vehicle_unit.schedule(warehouse_1, SKU3_ID, 100, 3)

    #     self.assertEqual(100, vehicle_unit.requested_quantity)
    #     self.assertEqual(3, vehicle_unit._remaining_patient)

    #     # no payload
    #     self.assertEqual(0, vehicle_unit.payload)
    #     self.assertEqual(0, vehicle_unit.data_model.payload)

    #     # push env to next step
    #     env.step(None)

    #     self.assertEqual(3 - 1, vehicle_unit._remaining_patient)
    #     self.assertEqual(0, vehicle_unit.payload)
    #     self.assertEqual(0, vehicle_unit.data_model.payload)

    #     # push env to next step
    #     env.step(None)

    #     self.assertEqual(3 - 1 - 1, vehicle_unit._remaining_patient)
    #     self.assertEqual(0, vehicle_unit.payload)
    #     self.assertEqual(0, vehicle_unit.data_model.payload)

    #     env.step(None)

    #     # Reset to max patient in reset function
    #     self.assertEqual(vehicle_unit._max_patient, vehicle_unit._remaining_patient)

    #     self.assertEqual(0, vehicle_unit.payload)
    #     self.assertEqual(0, vehicle_unit.data_model.payload)

    #     # ############################### ######################################
    #     states = vehicle_nodes[:vehicle_unit.data_model_index:features[IDX_PAYLOAD]].flatten().astype(np.int)

    #     # no payload from start to now
    #     self.assertListEqual([0] * 3, list(states))

    #     # ############################### ######################################
    #     # The vehicle will be reset to initial state once it arrives at the destination and unloads products.
    #     self.assertIsNone(vehicle_unit._destination)
    #     self.assertEqual(0, vehicle_unit.product_id)
    #     self.assertEqual(0, vehicle_unit._remaining_steps)
    #     self.assertEqual(0, vehicle_unit.payload)
    #     self.assertEqual(0, vehicle_unit._steps)
    #     self.assertEqual(0, vehicle_unit.requested_quantity)

    #     # check states
    #     states = vehicle_nodes[env.frame_index:vehicle_unit.data_model_index:features].flatten().astype(np.int)
    #     self.assertEqual(0, states[IDX_PAYLOAD])

    # def test_vehicle_unit_cannot_unload_at_destination(self) -> None:
    #     """Test Vehicle can not unload by scheduling more than Warehouse_001's remaining space from Supplier_SKU3.
    #     NOTE: If vehicle cannot unload at destination, it will keep waiting, until success to unload.
    #     """
    #     env = build_env("case_02", 10)
    #     be = env.business_engine
    #     assert isinstance(be, SupplyChainBusinessEngine)

    #     supplier_3 = be.world._get_facility_by_name("Supplier_SKU3")
    #     warehouse_1 = be.world._get_facility_by_name("Warehouse_001")

    #     vehicle_unit: VehicleUnit = supplier_3.distribution.children[0]
    #     self.assertTrue(isinstance(vehicle_unit, VehicleUnit))

    #     # make sure the upstream in the only one supplier in config
    #     self.assertEqual(1, len(warehouse_1.upstream_vlt_infos))
    #     self.assertEqual(1, len(warehouse_1.upstream_vlt_infos[SKU3_ID]))

    #     vehicle_nodes = env.snapshot_list["vehicle"]

    #     features = ("id", "facility_id", "payload", "unit_transport_cost")
    #     IDX_ID, IDX_FACILITY_ID, IDX_PAYLOAD, IDX_UNIT_TRANSPORT_COST = 0, 1, 2, 3

    #     # Move all 80 sku3 to Warehouse_001 (only 70 remaining space), will cause vehicle keep waiting there
    #     vehicle_unit.schedule(warehouse_1, SKU3_ID, 80, 3)

    #     # step to the end.
    #     is_done = False

    #     while not is_done:
    #         _, _, is_done = env.step(None)

    #     # Payload should be 80 for first 2 ticks, as it is on the way.
    #     # Payload of tick 3 should be 10 as it will unload (100 - 10 * 3 = 70) as soon as it arrives at the destination.
    #     # Later there is no storage space left to unload more products, so it would be 10 until the end.
    #     states = vehicle_nodes[:vehicle_unit.data_model_index:features[IDX_PAYLOAD]].flatten().astype(np.int)
    #     self.assertListEqual([80] * 2 + [10] * (10 - 2), list(states))

    """
    Distribution unit test:

    . initial state
    . place order
    . dispatch orders without available vehicle
    . dispatch order with vehicle
    """

    def test_distribution_unit_initial_state(self) -> None:
        """Test initial state of the DistributionUnit of Supplier_SKU3."""
        env = build_env("case_02", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        supplier_3 = be.world._get_facility_by_name("Supplier_SKU3")
        distribution_unit = supplier_3.distribution

        self.assertEqual(0, distribution_unit.delay_order_penalty[SKU3_ID])
        self.assertEqual(0, distribution_unit.check_in_quantity_in_order[SKU3_ID])
        self.assertEqual(0, distribution_unit.transportation_cost[SKU3_ID])
        self.assertEqual(10, distribution_unit._unit_delay_order_penalty[SKU3_ID])

        self.assertEqual(0, sum([len(order_queue) for order_queue in distribution_unit._order_queues.values()]))

        env.reset()

        self.assertEqual(0, sum([len(order_queue) for order_queue in distribution_unit._order_queues.values()]))

    def test_distribution_unit_dispatch_order(self) -> None:
        """Test initial state of the DistributionUnit of Supplier_SKU3."""
        env = build_env("case_02", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        supplier_3 = be.world._get_facility_by_name("Supplier_SKU3")
        warehouse_1 = be.world._get_facility_by_name("Warehouse_001")

        distribution_unit = supplier_3.distribution
        product_unit = supplier_3.products[SKU3_ID]

        vehicle_1: Vehicle = distribution_unit.vehicles["train"][0]
        vehicle_2: Vehicle = distribution_unit.vehicles["train"][1]

        order = Order(warehouse_1, SKU3_ID, 10, "train", 7)
        distribution_unit.place_order(order)

        # Check if the order already saved in queue
        self.assertEqual(1, len(distribution_unit._order_queues["train"]))
        # check get pending order correct
        self.assertEqual(10, sum([order.quantity for order in distribution_unit._order_queues["train"]]))

        # same as vehicle schedule case, distribution will try to schedule this order to vehicles from beginning to end
        # so it will dispatch this order to first vehicle
        env.step(None)

        self.assertEqual(warehouse_1, vehicle_1._destination)
        self.assertEqual(10, vehicle_1.requested_quantity)
        self.assertEqual(SKU3_ID, vehicle_1.product_id)

        # since we already test vehicle unit, do not check the it again here

        # add another order to check pending order
        distribution_unit.place_order(order)

        # 1 pending order now
        self.assertEqual(1, len(distribution_unit._order_queues["train"]))
        self.assertEqual(10, sum([order.quantity for order in distribution_unit._order_queues["train"]]))

        # another order, will cause the pending order increase
        distribution_unit.place_order(order)

        # 2 pending orders now
        self.assertEqual(2, len(distribution_unit._order_queues["train"]))
        self.assertEqual(10 * 2, sum([order.quantity for order in distribution_unit._order_queues["train"]]))

        # now we have only one available vehicle, 2 pending order
        # next step will cause delay_order_penalty
        env.step(None)

        self.assertEqual(warehouse_1, vehicle_2._destination)
        self.assertEqual(10, vehicle_2.requested_quantity)
        self.assertEqual(SKU3_ID, vehicle_2.product_id)

        # Only 1 pending order left
        self.assertEqual(1, len(distribution_unit._order_queues["train"]))
        self.assertEqual(10, sum([order.quantity for order in distribution_unit._order_queues["train"]]))

        # NOTE: the delay order penalty would be set to 0 by ProductUnit.
        self.assertEqual(10, product_unit._delay_order_penalty)

        distribution_unit.place_order(order)
        distribution_unit.place_order(order)
        env.step(None)

        # NOTE: the delay order penalty would be set to 0 by ProductUnit.
        self.assertEqual(10 * (1 + 2), product_unit._delay_order_penalty)

    """
    Seller unit test:
        . initial state
        . with a customized seller unit
        . with built in one
    """

    def test_seller_unit_initial_states(self) -> None:
        """Test the initial states of sku3's SellerUnit of Retailer_001."""
        env = build_env("case_02", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        retailer_1 = be.world._get_facility_by_name("Retailer_001")
        seller_unit = retailer_1.products[SKU3_ID].seller

        self.assertEqual(SKU3_ID, seller_unit.product_id)

        # from configuration
        self.assertEqual(10, seller_unit._gamma)

        # initial value
        self.assertEqual(0, seller_unit._sold)
        self.assertEqual(0, seller_unit._demand)
        self.assertEqual(0, seller_unit._total_sold)

        # initial data model value
        self.assertEqual(seller_unit._sold, seller_unit.data_model.sold)
        self.assertEqual(seller_unit._demand, seller_unit.data_model.demand)
        self.assertEqual(seller_unit._total_sold, seller_unit.data_model.total_sold)

        env.reset()

        self.assertEqual(SKU3_ID, seller_unit.product_id)

        # from configuration
        self.assertEqual(10, seller_unit._gamma)

        self.assertEqual(0, seller_unit._sold)
        self.assertEqual(0, seller_unit._demand)
        self.assertEqual(0, seller_unit._total_sold)

        self.assertEqual(seller_unit._sold, seller_unit.data_model.sold)
        self.assertEqual(seller_unit._demand, seller_unit.data_model.demand)
        self.assertEqual(seller_unit._total_sold, seller_unit.data_model.total_sold)

    def test_seller_unit_demand_states(self) -> None:
        """Test the demand states of sku3's SellerUnit of Retailer_001."""
        env = build_env("case_02", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        retailer_1 = be.world._get_facility_by_name("Retailer_001")
        seller_unit = retailer_1.products[SKU3_ID].seller

        seller_node_index = seller_unit.data_model_index

        seller_nodes = env.snapshot_list["seller"]

        features = ("sold", "demand", "total_sold")
        IDX_SOLD, IDX_DEMAND, IDX_TOTAL_SOLD = 0, 1, 2

        self.assertEqual(10, retailer_1.skus[SKU3_ID].init_stock)
        self.assertEqual(10, retailer_1.storage._product_level[SKU3_ID])
        init_stock = 10

        env.step(None)

        # seller unit will try to count down the product quantity based on demand
        # default seller use gamma distribution on each tick

        # demand should be same with original
        self.assertEqual(seller_unit._demand, seller_unit.data_model.demand)

        actual_sold = min(seller_unit._demand, init_stock)
        # sold may be not same as demand, depend on remaining quantity in storage
        self.assertEqual(actual_sold, seller_unit._sold)
        self.assertEqual(actual_sold, seller_unit.data_model.sold)
        self.assertEqual(actual_sold, seller_unit._total_sold)
        self.assertEqual(actual_sold, seller_unit.data_model.total_sold)

        states = seller_nodes[env.frame_index:seller_node_index:features].flatten().astype(np.int)

        self.assertEqual(actual_sold, states[IDX_SOLD])
        self.assertEqual(seller_unit._demand, states[IDX_DEMAND])
        self.assertEqual(actual_sold, states[IDX_TOTAL_SOLD])

        # move to next step to check if state is correct
        env.step(None)

        # demand should be same with original
        self.assertEqual(seller_unit._demand, seller_unit.data_model.demand)

        actual_sold_2 = min(seller_unit._demand, init_stock - actual_sold)

        # sold may be not same as demand, depend on remaining quantity in storage
        self.assertEqual(actual_sold_2, seller_unit._sold)
        self.assertEqual(actual_sold_2, seller_unit.data_model.sold)
        self.assertEqual(actual_sold + actual_sold_2, seller_unit._total_sold)
        self.assertEqual(actual_sold + actual_sold_2, seller_unit.data_model.total_sold)

        states = seller_nodes[env.frame_index:seller_node_index:features].flatten().astype(np.int)

        self.assertEqual(actual_sold_2, states[IDX_SOLD])
        self.assertEqual(seller_unit._demand, states[IDX_DEMAND])
        self.assertEqual(actual_sold + actual_sold_2, states[IDX_TOTAL_SOLD])

    def test_seller_unit_customized(self) -> None:
        """Test customized SellerUnit of sku3 in Retailer_001."""
        env = build_env("case_03", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        retailer_1 = be.world._get_facility_by_name("Retailer_001")
        seller_unit = retailer_1.products[SKU3_ID].seller

        seller_node_index = seller_unit.data_model_index

        seller_nodes = env.snapshot_list["seller"]

        features = ("sold", "demand", "total_sold")
        IDX_SOLD, IDX_DEMAND, IDX_TOTAL_SOLD = 0, 1, 2

        self.assertEqual(SKU3_ID, seller_unit.product_id)

        # NOTE: this simple seller unit return demands that same as current tick

        # ######################### Tick 0 ##############################
        env.step(None)

        # Tick 0 will have demand == 0
        # from configuration
        self.assertEqual(0, seller_unit._sold)
        self.assertEqual(0, seller_unit._demand)
        self.assertEqual(0, seller_unit._total_sold)

        self.assertEqual(0, seller_unit.data_model.sold)
        self.assertEqual(0, seller_unit.data_model.demand)
        self.assertEqual(0, seller_unit.data_model.total_sold)

        is_done = False
        while not is_done:
            _, _, is_done = env.step(None)

        states = seller_nodes[:seller_node_index:features[IDX_DEMAND]].flatten().astype(np.int)

        # Check demand history, it should be same as tick
        self.assertListEqual([i for i in range(100)], list(states))

        # Check sold states. Since the init stock 10 = 1 + 2 + 3 + 4, sold value should be 0 after tick 4.
        states = seller_nodes[:seller_node_index:features[IDX_SOLD]].flatten().astype(np.int)
        self.assertListEqual([0, 1, 2, 3, 4] + [0] * 95, list(states))

        # Check total sold, should be: 0, 0 + 1, 0 + 1 + 2, 0 + 1 + 2 + 3, 0 + 1 + 2 + 3 + 4, 10, 10, ...
        states = seller_nodes[:seller_node_index:features[IDX_TOTAL_SOLD]].flatten().astype(np.int)
        self.assertListEqual([0, 1, 3, 6, 10] + [10] * 95, list(states))

    """
    Units Interaction tests:
        . ComsumerUnit will receive products after vlt (+ 1) days
        . Order with 0-vlt
    """

    def test_consumer_receive_products_after_vlt_days(self) -> None:
        """Test Supplier_SKU1 ask products from Supplier_SKU3 and Supplier_SKU4 respectively.
        The Supplier_SKU3's DistributionUnit would be processed before Supplier_SKU1,
        so there would be vlt + 1 days before receiving,
        while Supplier_SKU4's DistributionUnit would be processed after Supplier_SKU1,
        so there would be only vlt days before Supplier_SKU1 receiving products from Supplier_SKU4.
        """
        env = build_env("case_01", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        env.step(None)

        supplier_1: FacilityBase = be.world._get_facility_by_name("Supplier_SKU1")
        supplier_3: FacilityBase = be.world._get_facility_by_name("Supplier_SKU3")
        supplier_4: FacilityBase = be.world._get_facility_by_name("Supplier_SKU4")
        sku3_consumer_unit = supplier_1.products[SKU3_ID].consumer
        consumer_node_index = sku3_consumer_unit.data_model_index

        features = ("id", "facility_id", "product_id", "order_base_cost", "purchased", "received", "order_product_cost")
        # IDX_ID, IDX_FACILITY_ID, IDX_PRODUCT_ID, IDX_ORDER_COST = 0, 1, 2, 3
        IDX_PURCHASED, IDX_RECEIVED, IDX_ORDER_PRODUCT_COST = 4, 5, 6

        consumer_nodes = env.snapshot_list["consumer"]

        # ############################## Ask products from Supplier_SKU3 #######################################

        required_quantity_1 = 1
        action = ConsumerAction(sku3_consumer_unit.id, SKU3_ID, supplier_3.id, required_quantity_1, "train")
        purchase_tick_1: int = env.tick

        # 1 day for scheduling according to the order of Supplier_SKU1 & Supplier_SKU3 in config
        # 7 days vlt, no extra days for loading and unloading
        expected_tick_1 = purchase_tick_1 + 1 + 7

        env.step([action])

        while env.tick < expected_tick_1 - 1:
            env.step(None)

        # Not received yet.
        self.assertEqual(required_quantity_1, sku3_consumer_unit._open_orders[supplier_3.id][SKU3_ID])
        self.assertEqual(0, sku3_consumer_unit._received)

        states = consumer_nodes[env.frame_index:consumer_node_index:features].flatten().astype(np.int)
        self.assertEqual(0, states[IDX_RECEIVED])

        env.step(None)

        self.assertEqual(expected_tick_1, env.tick)

        # now all order is done
        self.assertEqual(0, sku3_consumer_unit._open_orders[supplier_3.id][SKU3_ID])
        self.assertEqual(required_quantity_1, sku3_consumer_unit._received)

        states = consumer_nodes[env.frame_index:consumer_node_index:features].flatten().astype(np.int)
        self.assertEqual(required_quantity_1, states[IDX_RECEIVED])

        # ############################## Ask products from Supplier_SKU4 #######################################

        required_quantity_2 = 2
        action = ConsumerAction(sku3_consumer_unit.id, SKU3_ID, supplier_4.id, required_quantity_2, "train")
        purchase_tick_2: int = env.tick

        # 0 day for scheduling according to the order of Supplier_SKU1 & Supplier_SKU4 in config
        # 5 days vlt, no extra days for loading and unloading
        expected_tick_2 = purchase_tick_2 + 0 + 5

        env.step([action])

        while env.tick < expected_tick_2 - 1:
            env.step(None)

        # Not received yet.
        self.assertEqual(required_quantity_2, sku3_consumer_unit._open_orders[supplier_4.id][SKU3_ID])
        self.assertEqual(0, sku3_consumer_unit._received)

        states = consumer_nodes[env.frame_index:consumer_node_index:features].flatten().astype(np.int)
        self.assertEqual(0, states[IDX_RECEIVED])

        env.step(None)

        self.assertEqual(expected_tick_2, env.tick)

        # now all order is done
        self.assertEqual(0, sku3_consumer_unit._open_orders[supplier_4.id][SKU3_ID])
        self.assertEqual(required_quantity_2, sku3_consumer_unit._received)

        states = consumer_nodes[env.frame_index:consumer_node_index:features].flatten().astype(np.int)
        self.assertEqual(required_quantity_2, states[IDX_RECEIVED])

    # def test_0_vlt(self):
    #     """Test Supplier_SKU2 ask products from Supplier_SKU1 with 0-vlt."""
    #     env = build_env("case_01", 100)
    #     be = env.business_engine
    #     assert isinstance(be, SupplyChainBusinessEngine)

    #     env.step(None)

    #     supplier_1: FacilityBase = be.world._get_facility_by_name("Supplier_SKU1")
    #     supplier_2: FacilityBase = be.world._get_facility_by_name("Supplier_SKU2")
    #     sku1_consumer_unit = supplier_2.products[SKU1_ID].consumer
    #     consumer_node_index = sku1_consumer_unit.data_model_index

    #     features = ("id", "facility_id", "product_id", "order_base_cost", "purchased", "received", "order_product_cost")
    #     IDX_ID, IDX_FACILITY_ID, IDX_PRODUCT_ID, IDX_ORDER_COST = 0, 1, 2, 3
    #     IDX_PURCHASED, IDX_RECEIVED, IDX_ORDER_PRODUCT_COST = 4, 5 ,6

    #     consumer_nodes = env.snapshot_list["consumer"]

    #     # ############################## Ask products from Supplier_SKU3 #######################################

    #     required_quantity_1 = 10
    #     action = ConsumerAction(sku1_consumer_unit.id, SKU1_ID, supplier_1.id, required_quantity_1, "train")
    #     purchase_tick_1: int = env.tick

    #     # 1 day for scheduling according to the order of Supplier_SKU1 & Supplier_SKU2 in config
    #     # 0 days vlt, no extra days for loading and unloading
    #     expected_tick_1 = purchase_tick_1 + 1 + 0

    #     env.step([action])

    #     # TODO: Figure out 0-vlt case
    #     # self.assertEqual(expected_tick_1, env.tick)

    #     # TODO: Figure out 0-vlt case
    #     # now all order is done
    #     # self.assertEqual(required_quantity_1, sku1_consumer_unit._received)

    #     states = consumer_nodes[env.frame_index:consumer_node_index:features].flatten().astype(np.int)
    #     self.assertEqual(required_quantity_1, states[IDX_RECEIVED])


if __name__ == '__main__':
    unittest.main()
