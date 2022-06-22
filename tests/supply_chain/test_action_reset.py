# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
import unittest
from collections import defaultdict
from typing import Dict, List

import numpy as np

from maro.simulator.scenarios.supply_chain import (
    ConsumerAction,
    ConsumerUnit,
    FacilityBase,
    ManufactureAction,
    ManufactureUnit,
    RetailerFacility,
    StorageUnit,
    WarehouseFacility,
)
from maro.simulator.scenarios.supply_chain.business_engine import SupplyChainBusinessEngine

from tests.supply_chain.common import SKU1_ID, SKU3_ID, build_env, get_product_dict_from_storage


class MyTestCase(unittest.TestCase):
    """
    . consumer unit test
    . distribution unit test
    . manufacture unit test
    . seller unit test
    . storage unit test
    """

    def test_env_reset_with_none_action(self) -> None:
        """test_env_reset_with_none_action"""
        env = build_env("case_05", 500)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        env.step(None)

        supplier_1: FacilityBase = be.world._get_facility_by_name("Supplier_SKU1")
        warehouse_1 = be.world._get_facility_by_name("Warehouse_001")
        Store_001: FacilityBase = be.world._get_facility_by_name("Store_001")

        consumer_unit: ConsumerUnit = warehouse_1.products[SKU3_ID].consumer
        storage_unit: StorageUnit = supplier_1.storage
        Store_001.products[SKU3_ID].seller
        supplier_1.products[SKU1_ID].manufacture
        supplier_1.distribution

        consumer_nodes = env.snapshot_list["consumer"]
        storage_nodes = env.snapshot_list["storage"]
        seller_nodes = env.snapshot_list["seller"]
        manufacture_nodes = env.snapshot_list["manufacture"]
        distribution_nodes = env.snapshot_list["distribution"]

        consumer_features = (
            "id",
            "facility_id",
            "sku_id",
            "order_base_cost",
            "purchased",
            "received",
            "order_product_cost",
            "latest_consumptions",
            "in_transit_quantity",
        )

        storage_features = ("id", "facility_id")

        seller_features = (
            "sold",
            "demand",
            "total_sold",
            "id",
            "total_demand",
            "backlog_ratio",
            "facility_id",
            "product_unit_id",
        )

        manufacture_features = (
            "id",
            "facility_id",
            "start_manufacture_quantity",
            "sku_id",
            "in_pipeline_quantity",
            "finished_quantity",
            "product_unit_id",
        )

        distribution_features = ("id", "facility_id", "pending_order_number", "pending_product_quantity")

        # ##################################### Before reset #####################################

        expect_tick = 10

        # Save the env.metric of each tick into env_metric_1
        env_metric_1: Dict[int, dict] = defaultdict(dict)

        # Store the information about the snapshot of each tick in states_1_x
        states_1_consumer: Dict[int, dict] = defaultdict(dict)
        states_1_storage: Dict[int, dict] = defaultdict(dict)
        states_1_seller: Dict[int, dict] = defaultdict(dict)
        states_1_manufacture: Dict[int, dict] = defaultdict(dict)
        states_1_distribution: Dict[int, dict] = defaultdict(dict)

        for i in range(expect_tick):
            env.step(None)
            env_metric_1[i] = env.metrics

            for idx in range(len(consumer_nodes)):
                states_1_consumer[i][idx] = consumer_nodes[i:idx:consumer_features].flatten().astype(np.int)

            for idx in range(len(storage_nodes)):
                states_1_storage[i][idx] = list(storage_nodes[i:idx:storage_features].flatten().astype(np.int))
                states_1_storage[i][idx].append(storage_nodes[i:idx:"product_id_list"].flatten().astype(np.int))
                states_1_storage[i][idx].append(storage_nodes[i:idx:"product_quantity"].flatten().astype(np.int))
                states_1_storage[i][idx].append(storage_nodes[i:idx:"remaining_space"].flatten().astype(np.int))

            for idx in range(len(manufacture_nodes)):
                states_1_manufacture[i][idx] = manufacture_nodes[i:idx:manufacture_features].flatten().astype(np.int)

            for idx in range(len(distribution_nodes)):
                states_1_distribution[i][idx] = distribution_nodes[i:idx:distribution_features].flatten().astype(np.int)

            for idx in range(len(seller_nodes)):
                states_1_seller[i][idx] = seller_nodes[i:idx:seller_features].flatten().astype(np.int)

        # ############################### Test whether reset updates the storage unit completely ################
        env.reset()
        env.step(None)

        # snapshot should reset after env.reset().
        for idx in range(len(manufacture_nodes)):
            states = manufacture_nodes[1:idx:manufacture_features].flatten().astype(np.int)
            self.assertEqual([0, 0, 0, 0, 0, 0, 0], list(states))

        for idx in range(len(storage_nodes)):
            states = storage_nodes[1:idx:storage_features].flatten().astype(np.int)
            self.assertEqual([0, 0], list(states))

        for idx in range(len(distribution_nodes)):
            states = distribution_nodes[1:idx:distribution_features].flatten().astype(np.int)
            self.assertEqual([0, 0, 0, 0], list(states))

        for idx in range(len(consumer_nodes)):
            states = consumer_nodes[1:idx:consumer_features].flatten().astype(np.int)
            self.assertEqual([0, 0, 0, 0, 0, 0, 0, 0, 0], list(states))

        for idx in range(len(seller_nodes)):
            states = seller_nodes[1:idx:seller_features].flatten().astype(np.int)
            self.assertEqual([0, 0, 0, 0, 0, 0, 0, 0], list(states))

        expect_tick = 10

        # Save the env.metric of each tick into env_metric_2
        env_metric_2: Dict[int, dict] = defaultdict(dict)

        # Store the information about the snapshot storage unit of each tick in states_2

        states_2_consumer: Dict[int, dict] = defaultdict(dict)
        states_2_storage: Dict[int, dict] = defaultdict(dict)
        states_2_seller: Dict[int, dict] = defaultdict(dict)
        states_2_manufacture: Dict[int, dict] = defaultdict(dict)
        states_2_distribution: Dict[int, dict] = defaultdict(dict)

        for i in range(expect_tick):
            env.step(None)
            env_metric_2[i] = env.metrics

            for idx in range(len(consumer_nodes)):
                states_2_consumer[i][idx] = consumer_nodes[i:idx:consumer_features].flatten().astype(np.int)

            for idx in range(len(storage_nodes)):
                states_2_storage[i][idx] = list(storage_nodes[i:idx:storage_features].flatten().astype(np.int))
                states_2_storage[i][idx].append(storage_nodes[i:idx:"product_id_list"].flatten().astype(np.int))
                states_2_storage[i][idx].append(storage_nodes[i:idx:"product_quantity"].flatten().astype(np.int))
                states_2_storage[i][idx].append(storage_nodes[i:idx:"remaining_space"].flatten().astype(np.int))

            for idx in range(len(manufacture_nodes)):
                states_2_manufacture[i][idx] = manufacture_nodes[i:idx:manufacture_features].flatten().astype(np.int)

            for idx in range(len(distribution_nodes)):
                states_2_distribution[i][idx] = distribution_nodes[i:idx:distribution_features].flatten().astype(np.int)

            for idx in range(len(seller_nodes)):
                states_2_seller[i][idx] = seller_nodes[i:idx:seller_features].flatten().astype(np.int)

        for i in range(expect_tick):
            self.assertEqual(list(states_1_consumer[i]), list(states_2_consumer[i]))
            self.assertEqual(list(states_1_storage[i]), list(states_2_storage[i]))
            self.assertEqual(list(states_1_seller[i]), list(states_2_seller[i]))
            self.assertEqual(list(states_1_manufacture[i]), list(states_2_manufacture[i]))
            self.assertEqual(list(states_1_distribution[i]), list(states_2_distribution[i]))
            self.assertEqual(list(env_metric_1[i].values()), list(env_metric_2[i].values()))

    def test_env_reset_with_ManufactureAction_only(self) -> None:
        """test env reset with ManufactureAction only"""
        env = build_env("case_02", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        supplier_3: FacilityBase = be.world._get_facility_by_name("Supplier_SKU3")
        warehouse_1: WarehouseFacility = be.world._get_facility_by_name("Warehouse_001")
        retailer_1: RetailerFacility = be.world._get_facility_by_name("Retailer_001")

        storage_unit = supplier_3.storage
        warehouse_1.products[SKU3_ID].consumer
        manufacture_unit = supplier_3.products[SKU3_ID].manufacture
        supplier_3.distribution
        retailer_1.products[SKU3_ID].seller

        consumer_nodes = env.snapshot_list["consumer"]
        storage_nodes = env.snapshot_list["storage"]
        seller_nodes = env.snapshot_list["seller"]
        manufacture_nodes = env.snapshot_list["manufacture"]
        distribution_nodes = env.snapshot_list["distribution"]

        consumer_features = (
            "id",
            "facility_id",
            "sku_id",
            "order_base_cost",
            "purchased",
            "received",
            "order_product_cost",
            "latest_consumptions",
            "in_transit_quantity",
        )

        storage_features = ("id", "facility_id")

        seller_features = (
            "sold",
            "demand",
            "total_sold",
            "id",
            "total_demand",
            "backlog_ratio",
            "facility_id",
            "product_unit_id",
        )

        manufacture_features = (
            "id",
            "facility_id",
            "start_manufacture_quantity",
            "sku_id",
            "in_pipeline_quantity",
            "finished_quantity",
            "product_unit_id",
        )

        distribution_features = ("id", "facility_id", "pending_order_number", "pending_product_quantity")

        # ##################################### Before reset #####################################

        env.step(None)

        storage_node_index = storage_unit.data_model_index
        capacities = storage_nodes[env.frame_index : storage_node_index : "capacity"].flatten().astype(np.int)
        remaining_spaces = (
            storage_nodes[env.frame_index : storage_node_index : "remaining_space"].flatten().astype(np.int)
        )

        # there should be 80 units been taken at the beginning according to the config file.
        # so remaining space should be 20
        self.assertEqual(20, remaining_spaces.sum())
        # capacity is 100 by config
        self.assertEqual(100, capacities.sum())

        product_dict = get_product_dict_from_storage(env, env.frame_index, storage_node_index)

        # The product quantity should be same as configuration at beginning.
        # 80 sku3
        self.assertEqual(80, product_dict[SKU3_ID])

        ManufactureAction(manufacture_unit.id, 1)

        expect_tick = 30

        # Save the env.metric of each tick into env_metric_1
        env_metric_1: Dict[int, dict] = defaultdict(dict)

        # Store the information about the snapshot unit of each tick in states_1
        states_1_consumer: Dict[int, dict] = defaultdict(dict)
        states_1_storage: Dict[int, dict] = defaultdict(dict)
        states_1_seller: Dict[int, dict] = defaultdict(dict)
        states_1_manufacture: Dict[int, dict] = defaultdict(dict)
        states_1_distribution: Dict[int, dict] = defaultdict(dict)

        random_tick: List[int] = []

        # The purpose is to randomly perform the order operation
        for i in range(10):
            random_tick.append(random.randint(1, 30))

        for i in range(expect_tick):
            env.step([ManufactureAction(manufacture_unit.id, 1)])
            env_metric_1[i] = env.metrics

            for idx in range(len(consumer_nodes)):
                states_1_consumer[i][idx] = consumer_nodes[i:idx:consumer_features].flatten().astype(np.int)

            for idx in range(len(storage_nodes)):
                states_1_storage[i][idx] = list(storage_nodes[i:idx:storage_features].flatten().astype(np.int))
                states_1_storage[i][idx].append(storage_nodes[i:idx:"product_id_list"].flatten().astype(np.int))
                states_1_storage[i][idx].append(storage_nodes[i:idx:"product_quantity"].flatten().astype(np.int))
                states_1_storage[i][idx].append(storage_nodes[i:idx:"remaining_space"].flatten().astype(np.int))

            for idx in range(len(manufacture_nodes)):
                states_1_manufacture[i][idx] = manufacture_nodes[i:idx:manufacture_features].flatten().astype(np.int)

            for idx in range(len(distribution_nodes)):
                states_1_distribution[i][idx] = distribution_nodes[i:idx:distribution_features].flatten().astype(np.int)

            for idx in range(len(seller_nodes)):
                states_1_seller[i][idx] = seller_nodes[i:idx:seller_features].flatten().astype(np.int)

            if i in random_tick:
                env.step([ManufactureAction(manufacture_unit.id, 0)])

        # ############################### Test whether reset updates the manufacture unit completely ################
        env.reset()
        env.step(None)

        # snapshot should reset after env.reset().
        for idx in range(len(manufacture_nodes)):
            states = manufacture_nodes[1:idx:manufacture_features].flatten().astype(np.int)
            self.assertEqual([0, 0, 0, 0, 0, 0, 0], list(states))

        for idx in range(len(storage_nodes)):
            states = storage_nodes[1:idx:storage_features].flatten().astype(np.int)
            self.assertEqual([0, 0], list(states))

        for idx in range(len(distribution_nodes)):
            states = distribution_nodes[1:idx:distribution_features].flatten().astype(np.int)
            self.assertEqual([0, 0, 0, 0], list(states))

        for idx in range(len(consumer_nodes)):
            states = consumer_nodes[1:idx:consumer_features].flatten().astype(np.int)
            self.assertEqual([0, 0, 0, 0, 0, 0, 0, 0, 0], list(states))

        for idx in range(len(seller_nodes)):
            states = seller_nodes[1:idx:seller_features].flatten().astype(np.int)
            self.assertEqual([0, 0, 0, 0, 0, 0, 0, 0], list(states))

        capacities = storage_nodes[env.frame_index : storage_node_index : "capacity"].flatten().astype(np.int)
        remaining_spaces = (
            storage_nodes[env.frame_index : storage_node_index : "remaining_space"].flatten().astype(np.int)
        )

        # there should be 80 units been taken at the beginning according to the config file.
        # so remaining space should be 20
        self.assertEqual(20, remaining_spaces.sum())
        # capacity is 100 by config
        self.assertEqual(100, capacities.sum())

        product_dict = get_product_dict_from_storage(env, env.frame_index, storage_node_index)

        # The product quantity should be same as configuration at beginning.
        # 80 sku3
        self.assertEqual(80, product_dict[SKU3_ID])

        # all the id is greater than 0
        self.assertGreater(manufacture_unit.id, 0)

        expect_tick = 30

        # Save the env.metric of each tick into env_metric_2
        env_metric_2: Dict[int, dict] = defaultdict(dict)

        # Store the information about the snapshot unit of each tick in states_2

        states_2_consumer: Dict[int, dict] = defaultdict(dict)
        states_2_storage: Dict[int, dict] = defaultdict(dict)
        states_2_seller: Dict[int, dict] = defaultdict(dict)
        states_2_manufacture: Dict[int, dict] = defaultdict(dict)
        states_2_distribution: Dict[int, dict] = defaultdict(dict)

        for i in range(expect_tick):
            env.step(None)
            env_metric_2[i] = env.metrics

            for idx in range(len(consumer_nodes)):
                states_2_consumer[i][idx] = consumer_nodes[i:idx:consumer_features].flatten().astype(np.int)

            for idx in range(len(storage_nodes)):
                states_2_storage[i][idx] = list(storage_nodes[i:idx:storage_features].flatten().astype(np.int))
                states_2_storage[i][idx].append(storage_nodes[i:idx:"product_id_list"].flatten().astype(np.int))
                states_2_storage[i][idx].append(storage_nodes[i:idx:"product_quantity"].flatten().astype(np.int))
                states_2_storage[i][idx].append(storage_nodes[i:idx:"remaining_space"].flatten().astype(np.int))

            for idx in range(len(manufacture_nodes)):
                states_2_manufacture[i][idx] = manufacture_nodes[i:idx:manufacture_features].flatten().astype(np.int)

            for idx in range(len(distribution_nodes)):
                states_2_distribution[i][idx] = distribution_nodes[i:idx:distribution_features].flatten().astype(np.int)

            for idx in range(len(seller_nodes)):
                states_2_seller[i][idx] = seller_nodes[i:idx:seller_features].flatten().astype(np.int)

            if i in random_tick:
                env.step([ManufactureAction(manufacture_unit.id, 0)])

        expect_tick = 30
        for i in range(expect_tick):
            self.assertEqual(list(states_1_consumer[i]), list(states_2_consumer[i]))
            self.assertEqual(list(states_1_storage[i]), list(states_2_storage[i]))
            self.assertEqual(list(states_1_seller[i]), list(states_2_seller[i]))
            self.assertEqual(list(states_1_manufacture[i]), list(states_2_manufacture[i]))
            self.assertEqual(list(states_1_distribution[i]), list(states_2_distribution[i]))
            self.assertEqual(list(env_metric_1[i].values()), list(env_metric_2[i].values()))

    def test_env_reset_with_ConsumerAction_only(self) -> None:
        """ "test env reset with ConsumerAction only"""
        env = build_env("case_05", 500)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        env.step(None)

        warehouse_1 = be.world._get_facility_by_name("Warehouse_001")
        supplier_3: FacilityBase = be.world._get_facility_by_name("Supplier_SKU3")
        consumer_unit = warehouse_1.products[SKU3_ID].consumer

        consumer_nodes = env.snapshot_list["consumer"]
        storage_nodes = env.snapshot_list["storage"]
        seller_nodes = env.snapshot_list["seller"]
        manufacture_nodes = env.snapshot_list["manufacture"]
        distribution_nodes = env.snapshot_list["distribution"]

        consumer_features = (
            "id",
            "facility_id",
            "sku_id",
            "order_base_cost",
            "purchased",
            "received",
            "order_product_cost",
            "latest_consumptions",
            "in_transit_quantity",
        )

        storage_features = ("id", "facility_id")

        seller_features = (
            "sold",
            "demand",
            "total_sold",
            "id",
            "total_demand",
            "backlog_ratio",
            "facility_id",
            "product_unit_id",
        )

        manufacture_features = (
            "id",
            "facility_id",
            "start_manufacture_quantity",
            "sku_id",
            "in_pipeline_quantity",
            "finished_quantity",
            "product_unit_id",
        )

        distribution_features = ("id", "facility_id", "pending_order_number", "pending_product_quantity")

        # ##################################### Before reset #####################################
        action = ConsumerAction(consumer_unit.id, SKU3_ID, supplier_3.id, 1, "train")
        expect_tick = 100

        # Save the env.metric of each tick into env_metric_1
        env_metric_1: Dict[int, dict] = defaultdict(dict)

        # Store the information about the snapshot unit of each tick in states_1
        states_1_consumer: Dict[int, dict] = defaultdict(dict)
        states_1_storage: Dict[int, dict] = defaultdict(dict)
        states_1_seller: Dict[int, dict] = defaultdict(dict)
        states_1_manufacture: Dict[int, dict] = defaultdict(dict)
        states_1_distribution: Dict[int, dict] = defaultdict(dict)

        for i in range(expect_tick):
            env.step([action])
            env_metric_1[i] = env.metrics

            for idx in range(len(consumer_nodes)):
                states_1_consumer[i][idx] = consumer_nodes[i:idx:consumer_features].flatten().astype(np.int)

            for idx in range(len(storage_nodes)):
                states_1_storage[i][idx] = list(storage_nodes[i:idx:storage_features].flatten().astype(np.int))
                states_1_storage[i][idx].append(storage_nodes[i:idx:"product_id_list"].flatten().astype(np.int))
                states_1_storage[i][idx].append(storage_nodes[i:idx:"product_quantity"].flatten().astype(np.int))
                states_1_storage[i][idx].append(storage_nodes[i:idx:"remaining_space"].flatten().astype(np.int))

            for idx in range(len(manufacture_nodes)):
                states_1_manufacture[i][idx] = (
                    manufacture_nodes[i:idx:manufacture_features]
                    .flatten()
                    .astype(
                        np.int,
                    )
                )

            for idx in range(len(distribution_nodes)):
                states_1_distribution[i][idx] = (
                    distribution_nodes[i:idx:distribution_features]
                    .flatten()
                    .astype(
                        np.int,
                    )
                )

            for idx in range(len(seller_nodes)):
                states_1_seller[i][idx] = seller_nodes[i:idx:seller_features].flatten().astype(np.int)

        # ############### Test whether reset updates the consumer unit completely ################
        env.reset()
        env.step(None)

        # snapshot should reset after env.reset()
        for idx in range(len(manufacture_nodes)):
            states = manufacture_nodes[1:idx:manufacture_features].flatten().astype(np.int)
            self.assertEqual([0, 0, 0, 0, 0, 0, 0], list(states))

        for idx in range(len(storage_nodes)):
            states = storage_nodes[1:idx:storage_features].flatten().astype(np.int)
            self.assertEqual([0, 0], list(states))

        for idx in range(len(distribution_nodes)):
            states = distribution_nodes[1:idx:distribution_features].flatten().astype(np.int)
            self.assertEqual([0, 0, 0, 0], list(states))

        for idx in range(len(consumer_nodes)):
            states = consumer_nodes[1:idx:consumer_features].flatten().astype(np.int)
            self.assertEqual([0, 0, 0, 0, 0, 0, 0, 0, 0], list(states))

        for idx in range(len(seller_nodes)):
            states = seller_nodes[1:idx:seller_features].flatten().astype(np.int)
            self.assertEqual([0, 0, 0, 0, 0, 0, 0, 0], list(states))

        expect_tick = 100

        # Save the env.metric of each tick into env_metric_2
        env_metric_2: Dict[int, dict] = defaultdict(dict)

        # Store the information about the snapshot unit of each tick in states_2
        states_2_consumer: Dict[int, dict] = defaultdict(dict)
        states_2_storage: Dict[int, dict] = defaultdict(dict)
        states_2_seller: Dict[int, dict] = defaultdict(dict)
        states_2_manufacture: Dict[int, dict] = defaultdict(dict)
        states_2_distribution: Dict[int, dict] = defaultdict(dict)

        for i in range(expect_tick):
            env.step([action])
            env_metric_2[i] = env.metrics

            for idx in range(len(consumer_nodes)):
                states_2_consumer[i][idx] = consumer_nodes[i:idx:consumer_features].flatten().astype(np.int)

            for idx in range(len(storage_nodes)):
                states_2_storage[i][idx] = list(storage_nodes[i:idx:storage_features].flatten().astype(np.int))
                states_2_storage[i][idx].append(storage_nodes[i:idx:"product_id_list"].flatten().astype(np.int))
                states_2_storage[i][idx].append(storage_nodes[i:idx:"product_quantity"].flatten().astype(np.int))
                states_2_storage[i][idx].append(storage_nodes[i:idx:"remaining_space"].flatten().astype(np.int))

            for idx in range(len(manufacture_nodes)):
                states_2_manufacture[i][idx] = manufacture_nodes[i:idx:manufacture_features].flatten().astype(np.int)

            for idx in range(len(distribution_nodes)):
                states_2_distribution[i][idx] = distribution_nodes[i:idx:distribution_features].flatten().astype(np.int)

            for idx in range(len(seller_nodes)):
                states_2_seller[i][idx] = seller_nodes[i:idx:seller_features].flatten().astype(np.int)

        expect_tick = 100

        for i in range(expect_tick):
            self.assertEqual(list(states_1_consumer[i]), list(states_2_consumer[i]))
            self.assertEqual(list(states_1_storage[i]), list(states_2_storage[i]))
            self.assertEqual(list(states_1_seller[i]), list(states_2_seller[i]))
            self.assertEqual(list(states_1_manufacture[i]), list(states_2_manufacture[i]))
            self.assertEqual(list(states_1_distribution[i]), list(states_2_distribution[i]))
            self.assertEqual(list(env_metric_1[i].values()), list(env_metric_2[i].values()))

    def test_env_reset_with_both_ManufactureAction_and_ConsumerAction(self) -> None:
        """test env reset with both ManufactureAction and ConsumerAction"""
        env = build_env("case_05", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        env.step(None)

        supplier_1: FacilityBase = be.world._get_facility_by_name("Supplier_SKU1")
        supplier_3: FacilityBase = be.world._get_facility_by_name("Supplier_SKU3")
        warehouse_1: RetailerFacility = be.world._get_facility_by_name("Warehouse_001")
        consumer_unit: ConsumerUnit = warehouse_1.products[SKU1_ID].consumer
        manufacture_unit: ManufactureUnit = supplier_1.products[SKU1_ID].manufacture
        storage_unit: StorageUnit = supplier_1.storage

        consumer_node_index = consumer_unit.data_model_index
        manufacture_node_index = manufacture_unit.data_model_index
        storage_unit.data_model_index

        consumer_features = (
            "id",
            "facility_id",
            "sku_id",
            "order_base_cost",
            "purchased",
            "received",
            "order_product_cost",
            "latest_consumptions",
            "in_transit_quantity",
        )

        storage_features = ("id", "facility_id")

        seller_features = (
            "sold",
            "demand",
            "total_sold",
            "id",
            "total_demand",
            "backlog_ratio",
            "facility_id",
            "product_unit_id",
        )

        manufacture_features = (
            "id",
            "facility_id",
            "start_manufacture_quantity",
            "sku_id",
            "in_pipeline_quantity",
            "finished_quantity",
            "product_unit_id",
        )

        distribution_features = ("id", "facility_id", "pending_order_number", "pending_product_quantity")

        consumer_nodes = env.snapshot_list["consumer"]
        storage_nodes = env.snapshot_list["storage"]
        seller_nodes = env.snapshot_list["seller"]
        manufacture_nodes = env.snapshot_list["manufacture"]
        distribution_nodes = env.snapshot_list["distribution"]

        # ##################################### Before reset #####################################
        action_consumer = ConsumerAction(consumer_unit.id, SKU1_ID, supplier_1.id, 5, "train")
        action_manufacture = ManufactureAction(manufacture_unit.id, 1)

        expect_tick = 100

        # Save the env.metric of each tick into env_metric_1
        env_metric_1: Dict[int, dict] = defaultdict(dict)

        # Store the information about the snapshot unit of each tick in states_1
        states_1_consumer: Dict[int, dict] = defaultdict(dict)
        states_1_storage: Dict[int, dict] = defaultdict(dict)
        states_1_seller: Dict[int, dict] = defaultdict(dict)
        states_1_manufacture: Dict[int, dict] = defaultdict(dict)
        states_1_distribution: Dict[int, dict] = defaultdict(dict)

        random_tick: List[int] = []

        # The purpose is to randomly perform the order operation
        for i in range(30):
            random_tick.append(random.randint(0, 90))

        # Store the information about the snapshot unit of each tick in states_1
        states_1_consumer: Dict[int, dict] = defaultdict(dict)
        states_1_storage: Dict[int, dict] = defaultdict(dict)
        states_1_seller: Dict[int, dict] = defaultdict(dict)
        states_1_manufacture: Dict[int, dict] = defaultdict(dict)
        states_1_distribution: Dict[int, dict] = defaultdict(dict)

        for i in range(expect_tick):
            if i in random_tick:
                env.step([action_manufacture])
                continue

            env.step([action_consumer])
            env_metric_1[i] = env.metrics

            for idx in range(len(consumer_nodes)):
                states_1_consumer[i][idx] = consumer_nodes[i:idx:consumer_features].flatten().astype(np.int)

            for idx in range(len(storage_nodes)):
                states_1_storage[i][idx] = list(storage_nodes[i:idx:storage_features].flatten().astype(np.int))
                states_1_storage[i][idx].append(storage_nodes[i:idx:"product_id_list"].flatten().astype(np.int))
                states_1_storage[i][idx].append(storage_nodes[i:idx:"product_quantity"].flatten().astype(np.int))
                states_1_storage[i][idx].append(storage_nodes[i:idx:"remaining_space"].flatten().astype(np.int))

            for idx in range(len(manufacture_nodes)):
                states_1_manufacture[i][idx] = (
                    manufacture_nodes[i:idx:manufacture_features]
                    .flatten()
                    .astype(
                        np.int,
                    )
                )

            for idx in range(len(distribution_nodes)):
                states_1_distribution[i][idx] = (
                    distribution_nodes[i:idx:distribution_features]
                    .flatten()
                    .astype(
                        np.int,
                    )
                )

            for idx in range(len(seller_nodes)):
                states_1_seller[i][idx] = seller_nodes[i:idx:seller_features].flatten().astype(np.int)
        # ############### Test whether reset updates the consumer unit completely ################
        env.reset()
        env.step(None)

        # snapshot should reset after env.reset()
        consumer_states = consumer_nodes[1:consumer_node_index:consumer_features].flatten().astype(np.int)
        manufacture_states = manufacture_nodes[1:manufacture_node_index:manufacture_features].flatten().astype(np.int)
        self.assertEqual([0, 0, 0, 0, 0, 0, 0, 0, 0], list(consumer_states))
        self.assertEqual([0, 0, 0, 0, 0, 0, 0], list(manufacture_states))

        expect_tick = 100

        # Save the env.metric of each tick into env_metric_2
        env_metric_2: Dict[int, dict] = defaultdict(dict)

        # Store the information about the snapshot consumer unit of each tick in states_2
        states_2_consumer: Dict[int, dict] = defaultdict(dict)
        states_2_storage: Dict[int, dict] = defaultdict(dict)
        states_2_seller: Dict[int, dict] = defaultdict(dict)
        states_2_manufacture: Dict[int, dict] = defaultdict(dict)
        states_2_distribution: Dict[int, dict] = defaultdict(dict)

        for i in range(expect_tick):
            if i in random_tick:
                env.step([action_manufacture])
                continue

            env.step([action_consumer])
            env_metric_2[i] = env.metrics

            for idx in range(len(consumer_nodes)):
                states_2_consumer[i][idx] = consumer_nodes[i:idx:consumer_features].flatten().astype(np.int)

            for idx in range(len(storage_nodes)):
                states_2_storage[i][idx] = list(storage_nodes[i:idx:storage_features].flatten().astype(np.int))
                states_2_storage[i][idx].append(storage_nodes[i:idx:"product_id_list"].flatten().astype(np.int))
                states_2_storage[i][idx].append(storage_nodes[i:idx:"product_quantity"].flatten().astype(np.int))
                states_2_storage[i][idx].append(storage_nodes[i:idx:"remaining_space"].flatten().astype(np.int))

            for idx in range(len(manufacture_nodes)):
                states_2_manufacture[i][idx] = (
                    manufacture_nodes[i:idx:manufacture_features]
                    .flatten()
                    .astype(
                        np.int,
                    )
                )

            for idx in range(len(distribution_nodes)):
                states_2_distribution[i][idx] = (
                    distribution_nodes[i:idx:distribution_features]
                    .flatten()
                    .astype(
                        np.int,
                    )
                )

            for idx in range(len(seller_nodes)):
                states_2_seller[i][idx] = seller_nodes[i:idx:seller_features].flatten().astype(np.int)

        expect_tick = 100
        for i in range(expect_tick):
            for unit_id, unit in be.world.units.items():
                self.assertEqual(list(states_1_consumer[i]), list(states_2_consumer[i]))
                self.assertEqual(list(states_1_storage[i]), list(states_2_storage[i]))
                self.assertEqual(list(states_1_seller[i]), list(states_2_seller[i]))
                self.assertEqual(list(states_1_manufacture[i]), list(states_2_manufacture[i]))
                self.assertEqual(list(states_1_distribution[i]), list(states_2_distribution[i]))
                self.assertEqual(list(env_metric_1[i].values()), list(env_metric_2[i].values()))


if __name__ == "__main__":
    unittest.main()
