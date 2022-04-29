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
from maro.simulator.scenarios.supply_chain.sku_dynamics_sampler import SkuDynamicsSampler, OneTimeSkuPriceDemandSampler, \
    OneTimeSkuDynamicsSampler, StreamSkuDynamicsSampler, StreamSkuPriceDemandSampler, DataFileDemandSampler
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
FOOD_1_ID = 20
HOBBY_1_ID = 30


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
            "id", "facility_id", "start_manufacture_quantity", "product_id",
        )
        IDX_ID, IDX_FACILITY_ID, IDX_START_MANUFACTURE_QUANTITY, IDX_PRODUCT_ID = 0, 1, 2, 3

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

        # let is generate 20, but actually it can only procedure 19 because the storage will reach the limitation.
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
        This sku supplier does not have enough source material at the begining,
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
            "id", "facility_id", "start_manufacture_quantity", "product_id"
        )
        IDX_ID, IDX_FACILITY_ID, IDX_START_MANUFACTURE_QUANTITY, IDX_PRODUCT_ID = 0, 1, 2, 3

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
        self.assertEqual(4, manufacture_states[IDX_PRODUCT_ID])

        product_dict = get_product_dict_from_storage(env, env.frame_index, sku4_storage_index)

        # 50 sku4 at beginning
        self.assertEqual(50, product_dict[SKU4_ID])

        # 0 sku2
        self.assertEqual(0, product_dict[SKU2_ID])

        # ############################### TICK: 1 - end ######################################

        is_done = False

        while not is_done:
            # push to the end, the storage should not changed, no matter what production rate we give it.
            _, _, is_done = env.step([ManufactureAction(manufacture_sku4_unit.id, 10)])

        manufacture_states = manufacture_nodes[
            env.frame_index:manufacture_sku4_unit.data_model_index:manufacture_features
        ].flatten().astype(np.int)

        # manufacture_quantity should be 0
        self.assertEqual(0, manufacture_states[IDX_START_MANUFACTURE_QUANTITY])

        # output product id should be same as configured.
        self.assertEqual(SKU4_ID, manufacture_states[IDX_PRODUCT_ID])

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
            "id", "facility_id", "start_manufacture_quantity", "product_id"
        )
        IDX_ID, IDX_FACILITY_ID, IDX_START_MANUFACTURE_QUANTITY, IDX_PRODUCT_ID = 0, 1, 2, 3

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
            "id", "facility_id", "start_manufacture_quantity", "product_id",
        )
        IDX_ID, IDX_FACILITY_ID, IDX_START_MANUFACTURE_QUANTITY, IDX_PRODUCT_ID = 0, 1, 2, 3

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

        manufacture_sku3_unit = supplier_3.products[SKU3_ID].manufacture
        # Take 0 action to stop manufacturing.
        env.step([ManufactureAction(manufacture_sku3_unit.id, 0)])

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
        purchased_tick = env.tick
        env.step([action])

        purchased_frame = env.business_engine.frame_index(purchased_tick)
        states = consumer_nodes[purchased_frame:consumer_node_index:features].flatten().astype(np.int)
        self.assertEqual(action.quantity, states[IDX_PURCHASED])
        self.assertEqual(0, states[IDX_RECEIVED])
        self.assertEqual(action.product_id, states[IDX_PRODUCT_ID])

        arrival_tick = purchased_tick + 7
        while env.tick <= arrival_tick:
            env.step(None)

        arrival_frame = env.business_engine.frame_index(arrival_tick)
        states = consumer_nodes[arrival_frame:consumer_node_index:features].flatten().astype(np.int)
        self.assertEqual(action.quantity, states[IDX_RECEIVED])

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
    Distribution unit test:

    . initial state
    . place order and dispatch with available vehicle
    . place order but has no available vehicle
    . if arrive at destination within special vlt
    . if support for 0-vlt
    . try_unload if target storage cannot take all
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

        order = Order(warehouse_1, SKU3_ID, 10, "train")

        # There are 2 "train" in total, and 1 left after scheduling this order.
        distribution_unit.place_order(env.tick, order)
        self.assertEqual(0, len(distribution_unit._order_queues["train"]))
        self.assertEqual(0, sum([order.quantity for order in distribution_unit._order_queues["train"]]))

        # add another order, it would be successfully scheduled, but none available vehicle left now.
        distribution_unit.place_order(env.tick, order)
        self.assertEqual(0, len(distribution_unit._order_queues["train"]))
        self.assertEqual(0, sum([order.quantity for order in distribution_unit._order_queues["train"]]))

        start_tick = env.tick
        expected_tick = start_tick + 7  # vlt = 7

        # 3rd order, will cause the pending order increase
        distribution_unit.place_order(env.tick, order)
        self.assertEqual(1, len(distribution_unit._order_queues["train"]))
        self.assertEqual(10, sum([order.quantity for order in distribution_unit._order_queues["train"]]))

        while env.tick < expected_tick:
            env.step(None)

        # will arrive at the end of this tick, still on the way.
        assert env.tick == expected_tick
        self.assertEqual(1, len(distribution_unit._order_queues["train"]))
        self.assertEqual(10, sum([order.quantity for order in distribution_unit._order_queues["train"]]))

        self.assertEqual(10 * 1, distribution_unit.delay_order_penalty[SKU3_ID])
        self.assertEqual(1 * 10 * 2, distribution_unit.transportation_cost[SKU3_ID])

        env.step(None)

        self.assertEqual(0, len(distribution_unit._order_queues["train"]))
        self.assertEqual(0, sum([order.quantity for order in distribution_unit._order_queues["train"]]))

        self.assertEqual(0, distribution_unit.delay_order_penalty[SKU3_ID])
        self.assertEqual(1 * 10 * 1, distribution_unit.transportation_cost[SKU3_ID])

    def test_distribution_unit_with_0_vlt(self):
        """Test Supplier_SKU2 ask products from Supplier_SKU1 with 0-vlt."""
        env = build_env("case_01", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        env.step(None)

        supplier_1: FacilityBase = be.world._get_facility_by_name("Supplier_SKU1")
        supplier_2: FacilityBase = be.world._get_facility_by_name("Supplier_SKU2")
        sku1_consumer_unit = supplier_2.products[SKU1_ID].consumer
        consumer_node_index = sku1_consumer_unit.data_model_index

        features = ("id", "facility_id", "product_id", "order_base_cost", "purchased", "received", "order_product_cost")
        IDX_ID, IDX_FACILITY_ID, IDX_PRODUCT_ID, IDX_ORDER_COST = 0, 1, 2, 3
        IDX_PURCHASED, IDX_RECEIVED, IDX_ORDER_PRODUCT_COST = 4, 5 ,6

        consumer_nodes = env.snapshot_list["consumer"]

        # ############################## Ask products from Supplier_SKU3 #######################################

        required_quantity = 10
        action = ConsumerAction(sku1_consumer_unit.id, SKU1_ID, supplier_1.id, required_quantity, "train")
        purchase_tick: int = env.tick

        env.step([action])

        purchase_frame = env.business_engine.frame_index(purchase_tick)
        states = consumer_nodes[purchase_frame:consumer_node_index:features].flatten().astype(np.int)
        self.assertEqual(required_quantity, states[IDX_PURCHASED])
        # expected tick is equal to purchase tick since vlt = 0
        self.assertEqual(required_quantity, states[IDX_RECEIVED])

    def test_distribution_unit_cannot_unload_at_destination(self) -> None:
        """Test can not unload in one time by scheduling more than Warehouse_001's remaining space from Supplier_SKU3.
        NOTE: If vehicle cannot unload at destination, it will keep waiting, until success to unload.
        """
        env = build_env("case_02", 10)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        supplier_3 = be.world._get_facility_by_name("Supplier_SKU3")
        warehouse_1 = be.world._get_facility_by_name("Warehouse_001")
        distribution_unit = supplier_3.distribution
        warehouse_storage_unit = warehouse_1.storage

        env.step(None)

        start_tick = env.tick
        expected_tick = start_tick + 7
        order = Order(warehouse_1, SKU3_ID, 80, "train")
        distribution_unit.place_order(start_tick, order)

        while env.tick <= expected_tick:
            # Check the inventory level in target storage
            quantity = get_product_dict_from_storage(
                env, env.frame_index, warehouse_storage_unit.data_model_index
            )[SKU3_ID]

            self.assertEqual(10, quantity)

            # Check the payload in the distribution
            self.assertEqual(1, len(distribution_unit._payload_on_the_way[expected_tick]))
            self.assertEqual(warehouse_1, distribution_unit._payload_on_the_way[expected_tick][0].order.destination)
            self.assertEqual(80, distribution_unit._payload_on_the_way[expected_tick][0].payload)

            env.step(None)

        # 100 - 10 * 3 = 70 SKU3 would be unloaded as soon as it arrives at the destination.
        # Later there is no storage space left to unload more products,
        # so the inventory level would be 10 + 70 = 80, and the payload would be 10 until the end.
        is_done = False
        while not is_done:
            # Check the inventory level in target storage
            quantity = get_product_dict_from_storage(
                env, env.frame_index, warehouse_storage_unit.data_model_index
            )[SKU3_ID]

            self.assertEqual(10 + 70, quantity)

            # Check the payload in the distribution
            expected_tick = env.tick
            self.assertEqual(1, len(distribution_unit._payload_on_the_way[expected_tick]))
            self.assertEqual(warehouse_1, distribution_unit._payload_on_the_way[expected_tick][0].order.destination)
            self.assertEqual(80 - 70, distribution_unit._payload_on_the_way[expected_tick][0].payload)

            _, _, is_done = env.step(None)

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

        # 7 days vlt, no extra days for loading and unloading
        expected_tick_1 = purchase_tick_1 + 7

        env.step([action])

        while env.tick <= expected_tick_1:
            self.assertEqual(required_quantity_1, sku3_consumer_unit._open_orders[supplier_3.id][SKU3_ID])
            env.step(None)

        self.assertEqual(0, sku3_consumer_unit._open_orders[supplier_3.id][SKU3_ID])

        expected_frame = env.business_engine.frame_index(expected_tick_1)

        # Not received yet.
        states = consumer_nodes[expected_frame - 1:consumer_node_index:features].flatten().astype(np.int)
        self.assertEqual(0, states[IDX_RECEIVED])

        # received.
        states = consumer_nodes[expected_frame:consumer_node_index:features].flatten().astype(np.int)
        self.assertEqual(required_quantity_1, states[IDX_RECEIVED])

        # ############################## Ask products from Supplier_SKU4 #######################################

        required_quantity_2 = 2
        action = ConsumerAction(sku3_consumer_unit.id, SKU3_ID, supplier_4.id, required_quantity_2, "train")
        purchase_tick_2: int = env.tick

        # 5 days vlt, no extra days for loading and unloading
        expected_tick_2 = purchase_tick_2 + 5

        env.step([action])

        while env.tick <= expected_tick_2:
            self.assertEqual(required_quantity_2, sku3_consumer_unit._open_orders[supplier_4.id][SKU3_ID])
            env.step(None)

        self.assertEqual(0, sku3_consumer_unit._open_orders[supplier_4.id][SKU3_ID])

        expected_frame = env.business_engine.frame_index(expected_tick_2)

        # Not received yet.
        states = consumer_nodes[expected_frame - 1:consumer_node_index:features].flatten().astype(np.int)
        self.assertEqual(0, states[IDX_RECEIVED])

        # received.
        states = consumer_nodes[expected_frame:consumer_node_index:features].flatten().astype(np.int)
        self.assertEqual(required_quantity_2, states[IDX_RECEIVED])

    def test_init_sku_dynamics_OneTimeSkuPriceDemandSampler(self):
        """Test the reading of "store_001" SKU information of OneTimeSkuPriceDemandSampler."""
        env = build_env("case_04", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)
        Store_001: FacilityBase = be.world._get_facility_by_name("Store_001")
        configs = Store_001.configs
        world = be.world
        sku = OneTimeSkuPriceDemandSampler(configs, world)
        assert isinstance(sku, OneTimeSkuDynamicsSampler)
        # Store_ 001 has two SKUs.
        self.assertEqual(2, len(sku._cache[0]))
        # The price of the first item of SKU with product ID 20 is 43.11954545454545 and the demand is 25.
        self.assertEqual(43.11954545454545, sku._cache[0][FOOD_1_ID]['Price'])
        self.assertEqual(25, sku._cache[0][FOOD_1_ID]['Demand'])

        # The price of the first item of SKU with product ID 30 is 28.32 and the demand is 80.
        self.assertEqual(28.32, sku._cache[0][HOBBY_1_ID]['Price'])
        self.assertEqual(80, sku._cache[0][HOBBY_1_ID]['Demand'])

        # The price of the second item of SKU with product ID 20 is 43.632777777777775 and the demand is 41
        self.assertEqual(43.632777777777775, sku._cache[1][FOOD_1_ID]['Price'])
        self.assertEqual(41, sku._cache[1][FOOD_1_ID]['Demand'])

        # Test sample_price() method of onetimeskupricedemandsampler
        product_FOOD_1_price = sku.sample_price(20, FOOD_1_ID)
        self.assertEqual(43.117, product_FOOD_1_price)

        # Test sample_demand() method of onetimeskupricedemandsampler
        demand_FOOD_1 = sku.sample_demand(20, FOOD_1_ID)
        self.assertEqual(35, demand_FOOD_1)

        product_HOBBY_1_price = sku.sample_price(20, HOBBY_1_ID)
        demand_HOBBY_1_price = sku.sample_demand(20, HOBBY_1_ID)
        self.assertEqual(28.320000000000004, product_HOBBY_1_price)
        self.assertEqual(20, demand_HOBBY_1_price)

    def test_init_sku_dynamics_StreamSkuPriceDemandSampler(self):
        """Test the reading of "store_001" SKU information of StreamSkuPriceDemandSampler."""
        env = build_env("case_04", 600)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)
        Store_001: FacilityBase = be.world._get_facility_by_name("Store_001")
        configs = Store_001.configs
        world = be.world

        sku_stream = StreamSkuPriceDemandSampler(configs, world)
        assert isinstance(sku_stream, StreamSkuPriceDemandSampler)
        # Streamskupricedemandsamples inherits streamskudynamicssampler，one tick one day
        product_FOOD_1_price = sku_stream.sample_price(0, FOOD_1_ID)
        demand_FOOD_1 = sku_stream.sample_demand(0, FOOD_1_ID)
        self.assertEqual(43.11954545454545, product_FOOD_1_price)
        self.assertEqual(25, demand_FOOD_1)

        product_FOOD_1_price = sku_stream.sample_price(20, FOOD_1_ID)
        demand_FOOD_1 = sku_stream.sample_demand(20, FOOD_1_ID)
        self.assertEqual(43.117, product_FOOD_1_price)
        self.assertEqual(35, demand_FOOD_1)

        product_HOBBY_1_price = sku_stream.sample_price(0, HOBBY_1_ID)
        self.assertEqual(28.32, product_HOBBY_1_price)

    def test_init_sku_dynamics_DataFileDemandSampler(self):
        """Test the reading of "store_001" SKU information of DataFileDemandSampler."""
        env = build_env("case_04", 600)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)
        Store_001: FacilityBase = be.world._get_facility_by_name("Store_001")
        configs = Store_001.configs
        world = be.world
        sku_datafile = DataFileDemandSampler(configs, world)
        demand_FOOD_1 = sku_datafile.sample_demand(20, FOOD_1_ID)
        self.assertEqual(35, demand_FOOD_1)

    def test__sku_dynamics_DataFileDemandSampler(self):
        """Tested the store_ 001 storage_ Interaction between unit and data."""
        env = build_env("case_04", 600)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        env.step(None)
        Store_001: FacilityBase = be.world._get_facility_by_name("Store_001")
        configs = Store_001.configs
        world = be.world
        sku_datafile = DataFileDemandSampler(configs, world)
        product_FOOD_1_price = sku_datafile.sample_demand(20, FOOD_1_ID)
        self.assertEqual(35, product_FOOD_1_price)

        storage_unit: StorageUnit = Store_001.storage
        storage_node_index = storage_unit.data_model_index

        storage_nodes = env.snapshot_list["storage"]

        # ######################### Product Quantity ###########################
        init_product_dict = get_product_dict_from_storage(env, env.frame_index, storage_node_index)
        self.assertEqual(2, len(init_product_dict))

        # Inside StorageUnit
        self.assertEqual(10000 - 25, storage_unit._product_level[FOOD_1_ID])
        self.assertEqual(5000 - 80, storage_unit._product_level[HOBBY_1_ID])
        # In Snapshot
        self.assertEqual(9975, init_product_dict[FOOD_1_ID])
        self.assertEqual(4920, init_product_dict[HOBBY_1_ID])

        # ######################### Capacity ###########################
        capacities = storage_nodes[env.frame_index:storage_node_index:"capacity"].flatten().astype(np.int)
        self.assertEqual(80000, storage_unit.capacity)
        self.assertEqual(80000, capacities.sum())

        # ######################### Remaining Space ###########################
        init_remaining_spaces = storage_nodes[env.frame_index:storage_node_index:"remaining_space"].flatten().astype(
            np.int)
        self.assertEqual(80000 - (10000 - 25) - (5000 - 80), init_remaining_spaces.sum())
        # ######################### tick 1 ###########################
        env.step(None)

        product_dict = get_product_dict_from_storage(env, env.frame_index, storage_node_index)
        self.assertEqual(2, len(product_dict))

        # Inside StorageUnit
        self.assertEqual(10000 - 25 - 41, storage_unit._product_level[FOOD_1_ID])
        self.assertEqual(5000 - 80 - 32, storage_unit._product_level[HOBBY_1_ID])

        expected_tick = 20
        while env.tick < expected_tick - 1:
            env.step(None)

        # ######################### Product Quantity ###########################
        product_dict = get_product_dict_from_storage(env, env.frame_index, storage_node_index)
        # Inside StorageUnit
        self.assertEqual(10000 - 915, storage_unit._product_level[FOOD_1_ID])
        self.assertEqual(5000 - 1186, storage_unit._product_level[HOBBY_1_ID])
        # In Snapshot
        self.assertEqual(10000 - 915, product_dict[FOOD_1_ID])
        self.assertEqual(5000 - 1186, product_dict[HOBBY_1_ID])

        # ######################### Capacity ###########################
        capacities = storage_nodes[env.frame_index:storage_node_index:"capacity"].flatten().astype(np.int)
        self.assertEqual(80000, storage_unit.capacity)
        self.assertEqual(80000, capacities.sum())

        # ######################### Remaining Space ###########################
        init_remaining_spaces = storage_nodes[
                                env.frame_index:storage_node_index:"remaining_space"
                                ].flatten().astype(np.int)
        self.assertEqual(80000 - 9085 - 3814, storage_unit.remaining_space)
        self.assertEqual(80000 - 9085 - 3814, init_remaining_spaces.sum())

        # Should  change  after reset
        env.reset()
        env.step(None)

        # ######################### Product Quantity ###########################
        product_dict = get_product_dict_from_storage(env, env.frame_index, storage_node_index)

        # Inside StorageUnit
        self.assertEqual(10000 - 25, storage_unit._product_level[FOOD_1_ID])
        self.assertEqual(5000 - 80, storage_unit._product_level[HOBBY_1_ID])
        # In Snapshot
        self.assertEqual(10000 - 25, product_dict[FOOD_1_ID])
        self.assertEqual(5000 - 80, product_dict[HOBBY_1_ID])

        # ######################### Capacity ###########################
        capacities = storage_nodes[env.frame_index:storage_node_index:"capacity"].flatten().astype(np.int)
        self.assertEqual(80000, storage_unit.capacity)
        self.assertEqual(80000, capacities.sum())

        # ######################### Remaining Space ###########################
        init_remaining_spaces = storage_nodes[
                                env.frame_index:storage_node_index:"remaining_space"
                                ].flatten().astype(np.int)
        self.assertEqual(80000 - (10000 - 25) - (5000 - 80), storage_unit.remaining_space)
        self.assertEqual(80000 - (10000 - 25) - (5000 - 80), init_remaining_spaces.sum())

    def test_sku_dynamics_OneTimeSkuPriceDemandSampler(self):
        """Tested the store_ 001 storage_ Interaction between unit and data."""
        env = build_env("case_04", 600)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        env.step(None)
        Store_001: FacilityBase = be.world._get_facility_by_name("Store_001")
        configs = Store_001.configs
        world = be.world
        sku_onetime = OneTimeSkuPriceDemandSampler(configs, world)
        demand_FOOD_1 = sku_onetime.sample_demand(20, FOOD_1_ID)
        self.assertEqual(35, demand_FOOD_1)

        storage_unit: StorageUnit = Store_001.storage
        storage_node_index = storage_unit.data_model_index

        storage_nodes = env.snapshot_list["storage"]

        # ######################### Product Quantity ###########################
        init_product_dict = get_product_dict_from_storage(env, env.frame_index, storage_node_index)
        self.assertEqual(2, len(init_product_dict))

        # Inside StorageUnit
        self.assertEqual(10000 - 25, storage_unit._product_level[FOOD_1_ID])
        self.assertEqual(5000 - 80, storage_unit._product_level[HOBBY_1_ID])
        # In Snapshot
        self.assertEqual(9975, init_product_dict[FOOD_1_ID])
        self.assertEqual(4920, init_product_dict[HOBBY_1_ID])
        # In Class method
        self.assertEqual(9975, 10000 - sku_onetime.sample_demand(0, FOOD_1_ID))
        self.assertEqual(4920, 5000 - sku_onetime.sample_demand(0, HOBBY_1_ID))

        # ######################### Capacity ###########################
        capacities = storage_nodes[env.frame_index:storage_node_index:"capacity"].flatten().astype(np.int)
        self.assertEqual(80000, storage_unit.capacity)
        self.assertEqual(80000, capacities.sum())

        # ######################### Remaining Space ###########################
        init_remaining_spaces = storage_nodes[env.frame_index:storage_node_index:"remaining_space"].flatten().astype(
            np.int)
        self.assertEqual(80000 - (10000 - 25) - (5000 - 80), init_remaining_spaces.sum())

        env.step(None)

        product_dict = get_product_dict_from_storage(env, env.frame_index, storage_node_index)
        self.assertEqual(2, len(product_dict))

        # Inside StorageUnit
        self.assertEqual(10000 - 25 - 41, storage_unit._product_level[FOOD_1_ID])
        self.assertEqual(5000 - 80 - 32, storage_unit._product_level[HOBBY_1_ID])

        expected_tick = 20
        while env.tick < expected_tick - 1:
            env.step(None)

        # ######################### Product Quantity ###########################
        product_dict = get_product_dict_from_storage(env, env.frame_index, storage_node_index)
        # Inside StorageUnit
        self.assertEqual(10000 - 915, storage_unit._product_level[FOOD_1_ID])
        self.assertEqual(5000 - 1186, storage_unit._product_level[HOBBY_1_ID])
        # In Snapshot
        self.assertEqual(10000 - 915, product_dict[FOOD_1_ID])
        self.assertEqual(5000 - 1186, product_dict[HOBBY_1_ID])


        # ######################### Capacity ###########################
        capacities = storage_nodes[env.frame_index:storage_node_index:"capacity"].flatten().astype(np.int)
        self.assertEqual(80000, storage_unit.capacity)
        self.assertEqual(80000, capacities.sum())

        # ######################### Remaining Space ###########################
        init_remaining_spaces = storage_nodes[
                                env.frame_index:storage_node_index:"remaining_space"
                                ].flatten().astype(np.int)
        self.assertEqual(80000 - 9085 - 3814, storage_unit.remaining_space)
        self.assertEqual(80000 - 9085 - 3814, init_remaining_spaces.sum())

        # Should  change  after reset
        env.reset()
        env.step(None)

        # ######################### Product Quantity ###########################
        product_dict = get_product_dict_from_storage(env, env.frame_index, storage_node_index)

        # Inside StorageUnit
        self.assertEqual(10000 - 25, storage_unit._product_level[FOOD_1_ID])
        self.assertEqual(5000 - 80, storage_unit._product_level[HOBBY_1_ID])
        # In Snapshot
        self.assertEqual(10000 - 25, product_dict[FOOD_1_ID])
        self.assertEqual(5000 - 80, product_dict[HOBBY_1_ID])
        # In Class method
        self.assertEqual(9975, 10000 - sku_onetime.sample_demand(0, FOOD_1_ID))
        self.assertEqual(4920, 5000 - sku_onetime.sample_demand(0, HOBBY_1_ID))

        # ######################### Capacity ###########################
        capacities = storage_nodes[env.frame_index:storage_node_index:"capacity"].flatten().astype(np.int)
        self.assertEqual(80000, storage_unit.capacity)
        self.assertEqual(80000, capacities.sum())

        # ######################### Remaining Space ###########################
        init_remaining_spaces = storage_nodes[
                                env.frame_index:storage_node_index:"remaining_space"
                                ].flatten().astype(np.int)
        self.assertEqual(80000 - (10000 - 25) - (5000 - 80), storage_unit.remaining_space)
        self.assertEqual(80000 - (10000 - 25) - (5000 - 80), init_remaining_spaces.sum())


    def test_sku_dynamics_product_OneTimeSkuPriceDemandSampler(self):
        """Tested the store_ 001 storage_ Interaction between unit and data."""
        env = build_env("case_04", 600)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        env.step(None)
        Store_001: FacilityBase = be.world._get_facility_by_name("Store_001")
        configs = Store_001.configs
        world = be.world
        sku_onetime = OneTimeSkuPriceDemandSampler(configs, world)
        seller_unit = Store_001.products[FOOD_1_ID].seller

        seller_node_index = seller_unit.data_model_index

        seller_nodes = env.snapshot_list["seller"]

        features = ("sold", "demand", "total_sold")
        IDX_SOLD, IDX_DEMAND, IDX_TOTAL_SOLD = 0, 1, 2

        self.assertEqual(20, seller_unit.product_id)

        # NOTE: this simple seller unit return demands that same as current tick

        # Tick 0 will have demand == 25.first row of data after preprocessing data.
        # from sample_preprocessed.csv
        self.assertEqual(25, seller_unit._sold)
        self.assertEqual(25, seller_unit._demand)
        self.assertEqual(25, seller_unit._total_sold)

        self.assertEqual(25, seller_unit.data_model.sold)
        self.assertEqual(25, seller_unit.data_model.demand)
        self.assertEqual(25, seller_unit.data_model.total_sold)

        expected_tick = 5
        while env.tick < expected_tick - 1:
            env.step(None)

        states = seller_nodes[:seller_node_index:features[IDX_DEMAND]].flatten().astype(np.int)

        states = seller_nodes[:seller_node_index:features[IDX_SOLD]].flatten().astype(np.int)
        self.assertListEqual([25, 41, 40, 74, 57] , list(states))

    def test_consumer_dynamics_product_OneTimeSkuPriceDemandSampler(self):
        """Tested the store_ 001 sonsumer Interaction between unit and data."""
        env = build_env("case_04", 600)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)
        env.step(None)
        Store_001: FacilityBase = be.world._get_facility_by_name("Store_001")
        configs = Store_001.configs
        world = be.world
        sku_onetime = OneTimeSkuPriceDemandSampler(configs, world)
        FOOD_1_consumer_unit = Store_001.products[FOOD_1_ID].consumer

        consumer_node_index = FOOD_1_consumer_unit.data_model_index

        features = ("id", "facility_id", "product_id", "order_base_cost", "purchased", "received", "order_product_cost")
        IDX_ID, IDX_FACILITY_ID, IDX_PRODUCT_ID, IDX_ORDER_COST = 0, 1, 2, 3
        IDX_PURCHASED, IDX_RECEIVED, IDX_ORDER_PRODUCT_COST = 4, 5, 6

        consumer_nodes = env.snapshot_list["consumer"]

        # check initial state
        self.assertEqual(0, FOOD_1_consumer_unit._received)
        self.assertEqual(0, FOOD_1_consumer_unit._purchased)
        self.assertEqual(0, FOOD_1_consumer_unit._order_product_cost)

        # check data model state
        # order cost from configuration
        self.assertEqual(200, FOOD_1_consumer_unit._unit_order_cost)

        # NOTE: 0 is an invalid(initial) id
        self.assertEqual(0, FOOD_1_consumer_unit.data_model.purchased)
        self.assertEqual(0, FOOD_1_consumer_unit.data_model.received)
        self.assertEqual(0, FOOD_1_consumer_unit.data_model.order_product_cost)

        # check sources
        for source_facility_id in FOOD_1_consumer_unit.source_facility_id_list:
            source_facility: FacilityBase = be.world.get_facility_by_id(source_facility_id)

            # check if source facility contains the FOOD_1 config
            self.assertTrue(FOOD_1_ID in source_facility.skus)

        env.step(None)

        # check state
        states = consumer_nodes[env.frame_index:consumer_node_index:features].flatten().astype(np.int)

        self.assertEqual(FOOD_1_consumer_unit.id, states[IDX_ID])
        self.assertEqual(FOOD_1_consumer_unit.facility.id, states[IDX_FACILITY_ID])
        self.assertEqual(FOOD_1_ID, states[IDX_PRODUCT_ID])
        self.assertEqual(0, states[IDX_ORDER_COST])

        env.reset()
        env.step(None)

        states = consumer_nodes[env.frame_index:consumer_node_index:features].flatten().astype(np.int)

        # Nothing happened at tick 0, so most states will be 0
        self.assertEqual(0, states[IDX_PURCHASED])
        self.assertEqual(0, states[IDX_RECEIVED])
        self.assertEqual(0, states[IDX_ORDER_PRODUCT_COST])

        self.assertEqual(FOOD_1_consumer_unit.id, states[IDX_ID])
        self.assertEqual(FOOD_1_ID, states[IDX_PRODUCT_ID])

        """test_consumer_action"""

        features = ("id", "facility_id", "product_id", "order_base_cost", "purchased", "received", "order_product_cost")
        IDX_ID, IDX_FACILITY_ID, IDX_PRODUCT_ID, IDX_ORDER_COST = 0, 1, 2, 3
        IDX_PURCHASED, IDX_RECEIVED, IDX_ORDER_PRODUCT_COST = 4, 5, 6

        consumer_nodes = env.snapshot_list["consumer"]

        # ############################### Test Action with 0 quantity ######################################
        # zero quantity will be ignore
        action_with_zero = ConsumerAction(FOOD_1_consumer_unit.id,FOOD_1_ID, Store_001.id, 0, "train")
        env.step([action_with_zero])

        states = consumer_nodes[env.frame_index:consumer_node_index:features].flatten().astype(np.int)

        # Nothing happened at tick 0, at the action will be recorded
        self.assertEqual(action_with_zero.product_id, states[IDX_PRODUCT_ID])
        self.assertEqual(action_with_zero.quantity, states[IDX_PURCHASED])

        self.assertEqual(FOOD_1_consumer_unit.id, states[IDX_ID])
        self.assertEqual(FOOD_1_ID, states[IDX_PRODUCT_ID])

        # ############################### Test Action with positive quantity ######################################
        action = ConsumerAction(FOOD_1_consumer_unit.id, FOOD_1_ID, Store_001.id, 0, "train")
        env.step([action])

        self.assertEqual(action.quantity, FOOD_1_consumer_unit._purchased)
        self.assertEqual(0, FOOD_1_consumer_unit._received)

        states = consumer_nodes[env.frame_index:consumer_node_index:features].flatten().astype(np.int)

        # action field should be recorded
        self.assertEqual(action.product_id, states[IDX_PRODUCT_ID])

        self.assertEqual(action.quantity, states[IDX_PURCHASED])

        # no receives
        self.assertEqual(0, states[IDX_RECEIVED])


if __name__ == '__main__':
    unittest.main()
