# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest
import numpy as np

from maro.simulator.scenarios.supply_chain import FacilityBase, StorageUnit
from maro.simulator.scenarios.supply_chain.business_engine import SupplyChainBusinessEngine
from maro.simulator.scenarios.supply_chain.sku_dynamics_sampler import (
    OneTimeSkuPriceDemandSampler, OneTimeSkuDynamicsSampler, StreamSkuPriceDemandSampler, DataFileDemandSampler
)

from tests.supply_chain.common import (
    build_env, get_product_dict_from_storage, SKU1_ID, SKU2_ID, SKU3_ID, SKU4_ID, FOOD_1_ID, HOBBY_1_ID
)


class MyTestCase(unittest.TestCase):
    """
       read date test:
            . initial state
                . with snapshot
                . with storage unit
                . with dynamics sampler
                    . OneTimeSkuPriceDemandSampler
                    . StreamSkuPriceDemandSampler
                    . DataFileDemandSampler
       """

    def test_read_the_supplier_from_snapshot_sku_data(self) -> None:
        """This test mainly tests reading supplier from snapshot SKU data"""
        env = build_env("case_01", 400)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)
        env.step(None)
        # ############################### TICK: 0 ######################################

        # ############################### supplier_1 ######################################

        supplier_1: FacilityBase = be.world._get_facility_by_name("Supplier_SKU1")
        sku1_storage_index = supplier_1.storage.data_model_index
        manufacture_sku1_unit = supplier_1.products[SKU1_ID].manufacture

        storage_nodes = env.snapshot_list["storage"]

        # tick 0 passed, no product manufacturing, verified in above case, pass checking it here.

        capacities = storage_nodes[env.frame_index:sku1_storage_index:"capacity"].flatten().astype(np.int)
        remaining_spaces = storage_nodes[env.frame_index:sku1_storage_index:"remaining_space"].flatten().astype(np.int)

        # capacity is 200 by config
        self.assertEqual(200, capacities.sum())
        # there should be 100 + 96 units been taken at the beginning according to the config file.
        # so remaining space should be 200 - (100 + 96) = 4
        self.assertEqual(200 - (100 + 96), remaining_spaces.sum())
        product_dict = get_product_dict_from_storage(env, env.frame_index, sku1_storage_index)

        # The product quantity should be same as configuration at beginning.
        # 100 sku3, 96 sku1
        self.assertEqual(100, product_dict[SKU3_ID])
        self.assertEqual(96, product_dict[SKU1_ID])

        # all the id is greater than 0
        self.assertGreater(manufacture_sku1_unit.id, 0)

        # ############################### supplier_2 ######################################

        supplier_2: FacilityBase = be.world._get_facility_by_name("Supplier_SKU2")
        sku2_storage_index = supplier_2.storage.data_model_index
        manufacture_sku2_unit = supplier_2.products[SKU2_ID].manufacture

        storage_nodes = env.snapshot_list["storage"]

        # tick 0 passed, no product manufacturing.

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

        # ############################## supplier_3 ######################################

        supplier_3: FacilityBase = be.world._get_facility_by_name("Supplier_SKU3")
        sku3_storage_index = supplier_3.storage.data_model_index
        manufacture_sku3_unit = supplier_3.products[SKU3_ID].manufacture

        storage_nodes = env.snapshot_list["storage"]

        # tick 0 passed, no product manufacturing.

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

        # ############################### supplier_4 ######################################

        supplier_4: FacilityBase = be.world._get_facility_by_name("Supplier_SKU4")
        sku4_storage_index = supplier_4.storage.data_model_index
        manufacture_sku4_unit = supplier_4.products[SKU4_ID].manufacture

        storage_nodes = env.snapshot_list["storage"]

        manufacture_nodes = env.snapshot_list["manufacture"]
        manufacture_features = (
            "id", "facility_id", "start_manufacture_quantity", "sku_id"
        )
        IDX_ID, IDX_FACILITY_ID, IDX_START_MANUFACTURE_QUANTITY, IDX_SKU_ID = 0, 1, 2, 3

        # tick 0 passed, no product manufacturing.

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

    def test_read_the_supplier_from_storage_unit_sku_data(self) -> None:
        """This test reads supplier_sku data from storage unit"""
        env = build_env("case_01", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        env.step(None)

        # ############################## supplier_1 ######################################

        supplier_1: FacilityBase = be.world._get_facility_by_name("Supplier_SKU1")
        storage_unit: StorageUnit = supplier_1.storage

        # Product Quantity
        self.assertEqual(200, storage_unit.capacity)

        # Capacity
        self.assertEqual(200, storage_unit.capacity)

        # Remaining Space
        self.assertEqual(200 - 96 - 100, storage_unit.remaining_space)

        # Should not change even after reset
        env.reset()
        env.step(None)
        self.assertEqual(96, storage_unit._product_level[SKU1_ID])
        self.assertEqual(100, storage_unit._product_level[SKU3_ID])

        # Check the upper limit of commodity storage quantity
        self.assertEqual(100, storage_unit._storage_sku_upper_bound[0][SKU1_ID])
        self.assertEqual(100, storage_unit._storage_sku_upper_bound[0][SKU3_ID])

        # Capacity
        self.assertEqual(200, storage_unit.capacity)

        # Remaining Space
        self.assertEqual(200 - 96 - 100, storage_unit.remaining_space)

        # ############################## supplier_2 ######################################

        supplier_2 = be.world._get_facility_by_name("Supplier_SKU2")
        storage_unit = supplier_2.storage

        # Check if capacity correct
        self.assertEqual(1, len(storage_unit.config))
        self.assertEqual(200, storage_unit.capacity)

        # Check if product quantity correct
        self.assertEqual(50, storage_unit._product_level[SKU2_ID])
        self.assertEqual(50, storage_unit._product_level[SKU1_ID])

        # Check the upper limit of commodity storage quantity
        self.assertEqual(100, storage_unit._storage_sku_upper_bound[0][SKU2_ID])
        self.assertEqual(100, storage_unit._storage_sku_upper_bound[0][SKU1_ID])

        # Remaining space should be 200- 50 - 50
        self.assertEqual(200 - 50 - 50, storage_unit.remaining_space)

        # ############################## supplier_3 ######################################
        supplier_3: FacilityBase = be.world._get_facility_by_name("Supplier_SKU3")
        storage_unit: StorageUnit = supplier_3.storage

        # Check if remaining quantity correct
        self.assertEqual(80, storage_unit._product_level[SKU3_ID])

        # Check if capacity correct
        self.assertEqual(1, len(storage_unit.config))
        self.assertEqual(100, storage_unit.capacity)
        self.assertEqual(100, storage_unit._total_capacity)

        # Check the upper limit of commodity storage quantity
        self.assertEqual(100, storage_unit._storage_sku_upper_bound[0][SKU3_ID])

        # Remaining space should be 100 - 50 - 50
        self.assertEqual(100 - 80, storage_unit.remaining_space)

        # ############################## supplier_4 ######################################
        supplier_4 = be.world._get_facility_by_name("Supplier_SKU4")
        storage_unit = supplier_4.storage

        # Check if capacity correct
        self.assertEqual(1, len(storage_unit.config))
        self.assertEqual(200, storage_unit.capacity)

        # Check if product quantity correct
        self.assertEqual(0, storage_unit._product_level[SKU2_ID])
        self.assertEqual(50, storage_unit._product_level[SKU4_ID])
        self.assertEqual(50, storage_unit._product_level[SKU3_ID])

        # Check the upper limit of commodity storage quantity
        self.assertEqual(66, storage_unit._storage_sku_upper_bound[0][SKU2_ID])
        self.assertEqual(66, storage_unit._storage_sku_upper_bound[0][SKU4_ID])
        self.assertEqual(68, storage_unit._storage_sku_upper_bound[0][SKU3_ID])

        # Remaining space should be 200- 50 - 50 - 0
        self.assertEqual(200 - 50 - 50, storage_unit.remaining_space)

    def test_init_sku_dynamics_one_time_sku_price_demand_sampler(self):
        """Test the reading of store_001 SKU information of OneTimeSkuPriceDemandSampler.
           The data file of this test is test_case_ 04.csv"""
        env = build_env("case_04", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        Store_001: FacilityBase = be.world._get_facility_by_name("Store_001")
        configs = Store_001.configs
        world = be.world
        sku = OneTimeSkuPriceDemandSampler(configs, world)
        assert isinstance(sku, OneTimeSkuDynamicsSampler)
        # Store_001 has two SKUs.
        self.assertEqual(2, len(sku._cache[0]))
        # The price of the first item of SKU with product ID 20 is 43.11954545 and the demand is 10.
        self.assertEqual(43.0, sku._cache[0][FOOD_1_ID]['Price'])
        self.assertEqual(10, sku._cache[0][FOOD_1_ID]['Demand'])

        # The price of the first item of SKU with product ID 30 is 28.32 and the demand is 80.
        self.assertEqual(28.32, sku._cache[0][HOBBY_1_ID]['Price'])
        self.assertEqual(100, sku._cache[0][HOBBY_1_ID]['Demand'])

        # The price of the second item of SKU with product ID 20 is 43.63277778 and the demand is 41.
        self.assertEqual(43.1, sku._cache[1][FOOD_1_ID]['Price'])
        self.assertEqual(20, sku._cache[1][FOOD_1_ID]['Demand'])

        # Test sample_price() method of onetimeskupricedemandsampler.
        product_FOOD_1_price = sku.sample_price(4, FOOD_1_ID)
        self.assertEqual(43.4, product_FOOD_1_price)

        # Test sample_demand() method of onetimeskupricedemandsampler
        demand_FOOD_1 = sku.sample_demand(4, FOOD_1_ID)
        self.assertEqual(50, demand_FOOD_1)

        product_HOBBY_1_price = sku.sample_price(4, HOBBY_1_ID)
        demand_HOBBY_1_price = sku.sample_demand(4, HOBBY_1_ID)
        self.assertEqual(28.32, product_HOBBY_1_price)
        self.assertEqual(500, demand_HOBBY_1_price)

    def test_init_sku_dynamics_stream_sku_price_demand_sampler(self):
        """Test the reading of store_001 SKU information of StreamSkuPriceDemandSampler.
           The data file of this test is test_case_04.csv"""
        env = build_env("case_04", 600)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        Store_001: FacilityBase = be.world._get_facility_by_name("Store_001")
        configs = Store_001.configs
        world = be.world
        sku_stream = StreamSkuPriceDemandSampler(configs, world)
        assert isinstance(sku_stream, StreamSkuPriceDemandSampler)

        # Streamskupricedemandsamples inherits streamskudynamicssampler,one tick one day
        product_FOOD_1_price = sku_stream.sample_price(0, FOOD_1_ID)
        demand_FOOD_1 = sku_stream.sample_demand(0, FOOD_1_ID)
        self.assertEqual(43.0, product_FOOD_1_price)
        self.assertEqual(10, demand_FOOD_1)

        product_FOOD_1_price = sku_stream.sample_price(4, FOOD_1_ID)
        demand_FOOD_1 = sku_stream.sample_demand(4, FOOD_1_ID)
        self.assertEqual(43.4, product_FOOD_1_price)
        self.assertEqual(50, demand_FOOD_1)

        product_HOBBY_1_price = sku_stream.sample_price(0, HOBBY_1_ID)
        self.assertEqual(28.32, product_HOBBY_1_price)

    def test_init_sku_dynamics_data_file_demand_sampler(self):
        """Test the reading of store_001 SKU information of DataFileDemandSampler.
           The data file of this test is test_case_04.csv"""
        env = build_env("case_04", 600)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        Store_001: FacilityBase = be.world._get_facility_by_name("Store_001")
        configs = Store_001.configs
        world = be.world
        sku_datafile = DataFileDemandSampler(configs, world)
        demand_FOOD_1 = sku_datafile.sample_demand(4, FOOD_1_ID)
        self.assertEqual(50, demand_FOOD_1)

    def test_storage_unit_dynamics_data_file_demand_sampler(self):
        """Under the DataFileDemandSampler class,test the store between the storage unit and the dynamics CSV data
           interaction. The data file of this test is test_case_04.csv """
        env = build_env("case_04", 600)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        env.step(None)
        Store_001: FacilityBase = be.world._get_facility_by_name("Store_001")
        configs = Store_001.configs
        world = be.world
        sku_datafile = DataFileDemandSampler(configs, world)
        product_FOOD_1_price = sku_datafile.sample_demand(3, FOOD_1_ID)
        self.assertEqual(40, product_FOOD_1_price)

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

        # ######################### Capacity ###########################
        capacities = storage_nodes[env.frame_index:storage_node_index:"capacity"].flatten().astype(np.int)
        self.assertEqual(80000, storage_unit.capacity)
        self.assertEqual(80000, capacities.sum())

        # ######################### Remaining Space ###########################
        init_remaining_spaces = storage_nodes[env.frame_index:storage_node_index:"remaining_space"].flatten().astype(
            np.int)
        self.assertEqual(80000 - (10000 - 10) - (5000 - 100), init_remaining_spaces.sum())
        # ######################### tick 1 ###########################
        env.step(None)

        product_dict = get_product_dict_from_storage(env, env.frame_index, storage_node_index)
        self.assertEqual(2, len(product_dict))

        # Inside StorageUnit
        self.assertEqual(10000 - 10 - 20, storage_unit._product_level[FOOD_1_ID])
        self.assertEqual(5000 - 100 - 200, storage_unit._product_level[HOBBY_1_ID])

        expected_tick = 4
        while env.tick <= expected_tick - 1:
            env.step(None)

        # ######################### Product Quantity ###########################
        product_dict = get_product_dict_from_storage(env, env.frame_index, storage_node_index)
        # Inside StorageUnit
        self.assertEqual(10000 - (10 + 20 + 30 + 40 + 50), storage_unit._product_level[FOOD_1_ID])
        self.assertEqual(5000 - (100 + 200 + 300 + 400 + 500), storage_unit._product_level[HOBBY_1_ID])
        # In Snapshot
        self.assertEqual(10000 - (10 + 20 + 30 + 40 + 50), product_dict[FOOD_1_ID])
        self.assertEqual(5000 - (100 + 200 + 300 + 400 + 500), product_dict[HOBBY_1_ID])

        # ######################### Capacity ###########################
        capacities = storage_nodes[env.frame_index:storage_node_index:"capacity"].flatten().astype(np.int)
        self.assertEqual(80000, storage_unit.capacity)
        self.assertEqual(80000, capacities.sum())

        # ######################### Remaining Space ###########################
        init_remaining_spaces = storage_nodes[
                                env.frame_index:storage_node_index:"remaining_space"
                                ].flatten().astype(np.int)
        self.assertEqual(80000 - (10000 - (10 + 20 + 30 + 40 + 50)) - (5000 - (100 + 200 + 300 + 400 + 500)),
                         storage_unit.remaining_space)
        self.assertEqual(80000 - (10000 - (10 + 20 + 30 + 40 + 50)) - (5000 - (100 + 200 + 300 + 400 + 500)),
                         init_remaining_spaces.sum())

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
