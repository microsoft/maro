# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest
import numpy as np

from maro.simulator.scenarios.supply_chain import FacilityBase, StorageUnit, ManufactureAction
from maro.simulator.scenarios.supply_chain.business_engine import SupplyChainBusinessEngine
from maro.simulator.scenarios.supply_chain.sku_dynamics_sampler import OneTimeSkuPriceDemandSampler
from maro.simulator.scenarios.supply_chain.units.storage import AddStrategy

from tests.supply_chain.common import (
    build_env, get_product_dict_from_storage, SKU1_ID, SKU2_ID, SKU3_ID, FOOD_1_ID, HOBBY_1_ID
)


class MyTestCase(unittest.TestCase):
    """
    Storage test:

    . take available
        . enough
        . not enough
    . try to add products
        . meet whole storage capacity limitation
            . fail if all
            . not fail if all
        . enough space
    . try to take products
        . have enough
        . not enough
    . get product quantity
    . with dynamics sampler
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

        for sku_id, product_quantity in init_product_dict.items():
            self.assertEqual(product_quantity, storage_unit._product_level[sku_id])

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

        for sku_id, product_quantity in init_product_dict.items():
            self.assertEqual(product_quantity, storage_unit._product_level[sku_id])

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

    def test_storage_unit_dynamics_one_time_sku_price_demand_sampler(self):
        """Under the OneTimeSkuPriceDemandSampler class,test the store between the storage unit and the dynamics CSV
           data interaction. The data file of this test is test_case_04.csv """
        env = build_env("case_04", 600)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        env.step(None)
        Store_001: FacilityBase = be.world._get_facility_by_name("Store_001")
        configs = Store_001.configs
        world = be.world
        sku_onetime = OneTimeSkuPriceDemandSampler(configs, world)
        demand_FOOD_1 = sku_onetime.sample_demand(4, FOOD_1_ID)
        self.assertEqual(50, demand_FOOD_1)

        storage_unit: StorageUnit = Store_001.storage
        storage_node_index = storage_unit.data_model_index

        storage_nodes = env.snapshot_list["storage"]

        # ######################### Product Quantity ###########################
        init_product_dict = get_product_dict_from_storage(env, env.frame_index, storage_node_index)
        self.assertEqual(2, len(init_product_dict))

        # Inside StorageUnit
        self.assertEqual(10000 - 10, storage_unit._product_level[FOOD_1_ID])
        self.assertEqual(5000 - 100, storage_unit._product_level[HOBBY_1_ID])
        # In Snapshot
        self.assertEqual(9990, init_product_dict[FOOD_1_ID])
        self.assertEqual(4900, init_product_dict[HOBBY_1_ID])
        # In Class method
        self.assertEqual(9990, 10000 - sku_onetime.sample_demand(0, FOOD_1_ID))
        self.assertEqual(4900, 5000 - sku_onetime.sample_demand(0, HOBBY_1_ID))

        # ######################### Capacity ###########################
        capacities = storage_nodes[env.frame_index:storage_node_index:"capacity"].flatten().astype(np.int)
        self.assertEqual(80000, storage_unit.capacity)
        self.assertEqual(80000, capacities.sum())

        # ######################### Remaining Space ###########################
        init_remaining_spaces = storage_nodes[env.frame_index:storage_node_index:"remaining_space"].flatten().astype(
            np.int)
        self.assertEqual(80000 - (10000 - 10) - (5000 - 100), init_remaining_spaces.sum())

        env.step(None)

        product_dict = get_product_dict_from_storage(env, env.frame_index, storage_node_index)
        self.assertEqual(2, len(product_dict))

        # Inside StorageUnit
        self.assertEqual(10000 - 10 - 20, storage_unit._product_level[FOOD_1_ID])
        self.assertEqual(5000 - 100 - 200, storage_unit._product_level[HOBBY_1_ID])

        expected_tick = 4
        while env.tick < expected_tick - 1:
            env.step(None)

        # ######################### Product Quantity ###########################
        product_dict = get_product_dict_from_storage(env, env.frame_index, storage_node_index)
        # Inside StorageUnit
        self.assertEqual(10000 - 100, storage_unit._product_level[FOOD_1_ID])
        self.assertEqual(5000 - 1000, storage_unit._product_level[HOBBY_1_ID])
        # In Snapshot
        self.assertEqual(10000 - 100, product_dict[FOOD_1_ID])
        self.assertEqual(5000 - 1000, product_dict[HOBBY_1_ID])

        # ######################### Capacity ###########################
        capacities = storage_nodes[env.frame_index:storage_node_index:"capacity"].flatten().astype(np.int)
        self.assertEqual(80000, storage_unit.capacity)
        self.assertEqual(80000, capacities.sum())

        # ######################### Remaining Space ###########################
        init_remaining_spaces = storage_nodes[
                                env.frame_index:storage_node_index:"remaining_space"
                                ].flatten().astype(np.int)
        self.assertEqual(80000 - 9900 - 4000, storage_unit.remaining_space)
        self.assertEqual(80000 - 9900 - 4000, init_remaining_spaces.sum())

        # Should  change  after reset
        env.reset()
        env.step(None)

        # ######################### Product Quantity ###########################
        product_dict = get_product_dict_from_storage(env, env.frame_index, storage_node_index)

        # Inside StorageUnit
        self.assertEqual(10000 - 10, storage_unit._product_level[FOOD_1_ID])
        self.assertEqual(5000 - 100, storage_unit._product_level[HOBBY_1_ID])
        # In Snapshot
        self.assertEqual(10000 - 10, product_dict[FOOD_1_ID])
        self.assertEqual(5000 - 100, product_dict[HOBBY_1_ID])
        # In Class method
        self.assertEqual(9990, 10000 - sku_onetime.sample_demand(0, FOOD_1_ID))
        self.assertEqual(4900, 5000 - sku_onetime.sample_demand(0, HOBBY_1_ID))

        # ######################### Capacity ###########################
        capacities = storage_nodes[env.frame_index:storage_node_index:"capacity"].flatten().astype(np.int)
        self.assertEqual(80000, storage_unit.capacity)
        self.assertEqual(80000, capacities.sum())

        # ######################### Remaining Space ###########################
        init_remaining_spaces = storage_nodes[
                                env.frame_index:storage_node_index:"remaining_space"
                                ].flatten().astype(np.int)
        self.assertEqual(80000 - (10000 - 10) - (5000 - 100), storage_unit.remaining_space)
        self.assertEqual(80000 - (10000 - 10) - (5000 - 100), init_remaining_spaces.sum())


if __name__ == '__main__':
    unittest.main()
