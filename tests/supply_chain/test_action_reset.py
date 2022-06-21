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
    StorageUnit,
)
from maro.simulator.scenarios.supply_chain.business_engine import SupplyChainBusinessEngine
from maro.simulator.scenarios.supply_chain.order import Order

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
        consumer_unit: ConsumerUnit = supplier_1.products[SKU3_ID].consumer
        storage_unit: StorageUnit = supplier_1.storage
        seller_unit = Store_001.products[SKU3_ID].seller
        manufacture_unit = supplier_1.products[SKU3_ID].manufacture
        distribution_unit = supplier_1.distribution

        consumer_nodes = env.snapshot_list["consumer"]
        storage_nodes = env.snapshot_list["storage"]
        seller_nodes = env.snapshot_list["seller"]
        manufacture_nodes = env.snapshot_list["manufacture"]
        distribution_nodes = env.snapshot_list["distribution"]

        consumer_node_index = consumer_unit.data_model_index
        storage_node_index = storage_unit.data_model_index
        seller_node_index = seller_unit.data_model_index
        manufacture_node_index = manufacture_unit.data_model_index
        distribution_node_index = distribution_unit.data_model_index

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

        random_tick: List[int] = []

        # The purpose is to randomly perform the order operation
        for i in range(10):
            random_tick.append(random.randint(1, 30))

        # Store the information about the snapshot of each tick in states_1_x
        states_1_consumer: Dict[int, list] = defaultdict(list)
        states_1_storage: Dict[int, list] = defaultdict(list)
        states_1_seller: Dict[int, list] = defaultdict(list)
        states_1_manufacture: Dict[int, list] = defaultdict(list)
        states_1_distribution: Dict[int, list] = defaultdict(list)

        for i in range(expect_tick):
            env.step(None)
            if i in random_tick:
                order = Order(
                    src_facility=supplier_1,
                    dest_facility=warehouse_1,
                    sku_id=SKU3_ID,
                    quantity=10,
                    vehicle_type="train",
                    creation_tick=env.tick,
                    expected_finish_tick=env.tick + 7,
                )
                distribution_unit.place_order(order)
                distribution_unit.try_schedule_orders(env.tick)
            env_metric_1[i] = env.metrics
            states_1_consumer[i] = consumer_nodes[i:consumer_node_index:consumer_features].flatten().astype(np.int)
            states_1_manufacture[i] = (
                manufacture_nodes[i:manufacture_node_index:manufacture_features]
                .flatten()
                .astype(
                    np.int,
                )
            )
            env_metric_1[i] = env.metrics
            states_1_storage[i] = list(storage_nodes[i:storage_node_index:storage_features].flatten().astype(np.int))
            states_1_storage[i].append(
                storage_nodes[i:storage_node_index:"product_id_list"].flatten().astype(np.int).sum(),
            )
            states_1_storage[i].append(
                storage_nodes[i:storage_node_index:"product_quantity"].flatten().astype(np.int).sum(),
            )
            states_1_storage[i].append(
                storage_nodes[i:storage_node_index:"remaining_space"].flatten().astype(np.int).sum(),
            )
            states_1_seller[i] = seller_nodes[i:seller_node_index:seller_features].flatten().astype(np.int)
            states_1_manufacture[i] = (
                manufacture_nodes[i:manufacture_node_index:manufacture_features]
                .flatten()
                .astype(
                    np.int,
                )
            )
            states_1_distribution[i] = (
                distribution_nodes[i:distribution_node_index:distribution_features]
                .flatten()
                .astype(
                    np.int,
                )
            )

        # ############################### Test whether reset updates the storage unit completely ################
        env.reset()
        env.step(None)

        # snapshot should reset after env.reset().
        consumer_states = consumer_nodes[1:consumer_node_index:consumer_features].flatten().astype(np.int)
        storage_states = storage_nodes[1:storage_node_index:storage_features].flatten().astype(np.int)
        seller_states = seller_nodes[1:seller_node_index:seller_features].flatten().astype(np.int)
        manufacture_states = manufacture_nodes[1:manufacture_node_index:manufacture_features].flatten().astype(np.int)
        distribution_states = (
            distribution_nodes[1:distribution_node_index:distribution_features]
            .flatten()
            .astype(
                np.int,
            )
        )

        self.assertEqual([0, 0, 0, 0, 0, 0, 0, 0, 0], list(consumer_states))
        self.assertEqual([0, 0], list(storage_states))
        self.assertEqual([0, 0, 0, 0, 0, 0, 0, 0], list(seller_states))
        self.assertEqual([0, 0, 0, 0, 0, 0, 0], list(manufacture_states))
        self.assertEqual([0, 0, 0, 0], list(distribution_states))

        expect_tick = 10

        # Save the env.metric of each tick into env_metric_2
        env_metric_2: Dict[int, dict] = defaultdict(dict)

        # Store the information about the snapshot storage unit of each tick in states_2

        states_2_consumer: Dict[int, list] = defaultdict(list)
        states_2_storage: Dict[int, list] = defaultdict(list)
        states_2_seller: Dict[int, list] = defaultdict(list)
        states_2_manufacture: Dict[int, list] = defaultdict(list)
        states_2_distribution: Dict[int, list] = defaultdict(list)

        for i in range(expect_tick):
            env.step(None)
            if i in random_tick:
                order = Order(
                    src_facility=supplier_1,
                    dest_facility=warehouse_1,
                    sku_id=SKU3_ID,
                    quantity=10,
                    vehicle_type="train",
                    creation_tick=env.tick,
                    expected_finish_tick=env.tick + 7,
                )
                distribution_unit.place_order(order)
                distribution_unit.try_schedule_orders(env.tick)
            env_metric_2[i] = env.metrics
            states_2_consumer[i] = consumer_nodes[i:consumer_node_index:consumer_features].flatten().astype(np.int)
            states_2_storage[i] = list(storage_nodes[i:storage_node_index:storage_features].flatten().astype(np.int))
            states_2_storage[i].append(
                storage_nodes[i:storage_node_index:"product_id_list"].flatten().astype(np.int).sum(),
            )
            states_2_storage[i].append(
                storage_nodes[i:storage_node_index:"product_quantity"].flatten().astype(np.int).sum(),
            )
            states_2_storage[i].append(
                storage_nodes[i:storage_node_index:"remaining_space"].flatten().astype(np.int).sum(),
            )
            states_2_seller[i] = seller_nodes[i:seller_node_index:seller_features].flatten().astype(np.int)
            states_2_manufacture[i] = (
                manufacture_nodes[i:manufacture_node_index:manufacture_features]
                .flatten()
                .astype(
                    np.int,
                )
            )
            states_2_distribution[i] = (
                distribution_nodes[i:distribution_node_index:distribution_features].flatten().astype(np.int)
            )

        for i in range(expect_tick):
            self.assertEqual(list(states_1_consumer[i]), list(states_2_consumer[i]))
            self.assertEqual(list(states_1_storage[i]), list(states_2_storage[i]))
            self.assertEqual(list(states_1_seller[i]), list(states_2_seller[i]))
            self.assertEqual(list(states_1_manufacture[i]), list(states_2_manufacture[i]))
            self.assertEqual(list(states_1_distribution[i]), list(states_2_distribution[i]))
            self.assertEqual(list(env_metric_1[i].values()), list(env_metric_2[i].values()))

    def test_env_reset_with_ManufactureAction_only(self) -> None:
        """test env reset with ManufactureAction only"""
        env = build_env("case_01", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        supplier_3: FacilityBase = be.world._get_facility_by_name("Supplier_SKU3")
        sku3_storage_index = supplier_3.storage.data_model_index
        manufacture_sku3_unit = supplier_3.products[SKU3_ID].manufacture
        sku3_manufacture_index = manufacture_sku3_unit.data_model_index

        storage_nodes = env.snapshot_list["storage"]
        manufacture_nodes = env.snapshot_list["manufacture"]
        manufacture_features = (
            "id",
            "facility_id",
            "start_manufacture_quantity",
            "sku_id",
            "in_pipeline_quantity",
            "finished_quantity",
            "product_unit_id",
        )
        # ##################################### Before reset #####################################

        env.step(None)

        capacities = storage_nodes[env.frame_index : sku3_storage_index : "capacity"].flatten().astype(np.int)
        remaining_spaces = (
            storage_nodes[env.frame_index : sku3_storage_index : "remaining_space"].flatten().astype(np.int)
        )

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

        action = ManufactureAction(manufacture_sku3_unit.id, 1)

        expect_tick = 30

        # Save the env.metric of each tick into env_metric_1
        env_metric_1: Dict[int, dict] = defaultdict(dict)

        # Store the information about the snapshot manufacture unit of each tick in states_1
        states_1: Dict[int, list] = defaultdict(list)

        random_tick: List[int] = []

        # The purpose is to randomly perform the order operation
        for i in range(10):
            random_tick.append(random.randint(1, 30))

        for i in range(expect_tick):
            env.step([action])
            if i in random_tick:
                env.step([ManufactureAction(manufacture_sku3_unit.id, 1)])
            env_metric_1[i] = env.metrics
            states_1[i] = manufacture_nodes[i:sku3_manufacture_index:manufacture_features].flatten().astype(np.int)

        # ############################### Test whether reset updates the manufacture unit completely ################
        env.reset()
        env.step(None)

        # snapshot should reset after env.reset().
        states = manufacture_nodes[1:sku3_manufacture_index:manufacture_features].flatten().astype(np.int)
        self.assertEqual([0, 0, 0, 0, 0, 0, 0], list(states))

        storage_nodes = env.snapshot_list["storage"]
        manufacture_nodes = env.snapshot_list["manufacture"]

        capacities = storage_nodes[env.frame_index : sku3_storage_index : "capacity"].flatten().astype(np.int)
        remaining_spaces = (
            storage_nodes[env.frame_index : sku3_storage_index : "remaining_space"].flatten().astype(np.int)
        )

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

        expect_tick = 30

        # Save the env.metric of each tick into env_metric_2
        env_metric_2: Dict[int, dict] = defaultdict(dict)

        # Store the information about the snapshot manufacture unit of each tick in states_2
        states_2: Dict[int, list] = defaultdict(list)

        for i in range(expect_tick):
            env.step([action])
            if i in random_tick:
                env.step([ManufactureAction(manufacture_sku3_unit.id, 1)])
            env_metric_2[i] = env.metrics
            states_2[i] = manufacture_nodes[i:sku3_manufacture_index:manufacture_features].flatten().astype(np.int)

        expect_tick = 30
        for i in range(expect_tick):
            self.assertEqual(list(states_1[i]), list(states_2[i]))
            self.assertEqual(list(env_metric_1[i].values()), list(env_metric_2[i].values()))

    def test_env_reset_with_ConsumerAction_only(self) -> None:
        """ "test env reset with ConsumerAction only"""
        env = build_env("case_01", 500)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        env.step(None)

        supplier_1: FacilityBase = be.world._get_facility_by_name("Supplier_SKU1")
        supplier_3: FacilityBase = be.world._get_facility_by_name("Supplier_SKU3")
        sku3_consumer_unit = supplier_1.products[SKU3_ID].consumer

        consumer_node_index = sku3_consumer_unit.data_model_index

        features = (
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

        # ##################################### Before reset #####################################
        consumer_nodes = env.snapshot_list["consumer"]
        action = ConsumerAction(sku3_consumer_unit.id, SKU3_ID, supplier_3.id, 1, "train")
        expect_tick = 100

        # Save the env.metric of each tick into env_metric_1
        env_metric_1: Dict[int, dict] = defaultdict(dict)

        # Store the information about the snapshot consumer unit of each tick in states_1
        states_1: Dict[int, list] = defaultdict(list)

        for i in range(expect_tick):
            env.step([action])
            env_metric_1[i] = env.metrics
            states_1[i] = consumer_nodes[i:consumer_node_index:features].flatten().astype(np.int)

        # ############### Test whether reset updates the consumer unit completely ################
        env.reset()
        env.step(None)

        # snapshot should reset after env.reset()
        states = consumer_nodes[1:consumer_node_index:features].flatten().astype(np.int)
        self.assertEqual([0, 0, 0, 0, 0, 0, 0, 0, 0], list(states))

        expect_tick = 100

        # Save the env.metric of each tick into env_metric_2
        env_metric_2: Dict[int, dict] = defaultdict(dict)

        # Store the information about the snapshot consumer unit of each tick in states_2
        states_2: Dict[int, list] = defaultdict(list)
        for i in range(expect_tick):
            env.step([action])
            env_metric_2[i] = env.metrics
            states_2[i] = consumer_nodes[i:consumer_node_index:features].flatten().astype(np.int)

        expect_tick = 100
        for i in range(expect_tick):
            self.assertEqual(list(states_1[i]), list(states_2[i]))
            self.assertEqual(list(env_metric_1[i].values()), list(env_metric_2[i].values()))

    def test_env_reset_with_both_ManufactureAction_and_ConsumerAction(self) -> None:
        """test env reset with both ManufactureAction and ConsumerAction"""
        env = build_env("case_01", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        env.step(None)

        supplier_1: FacilityBase = be.world._get_facility_by_name("Supplier_SKU1")
        supplier_3: FacilityBase = be.world._get_facility_by_name("Supplier_SKU3")
        consumer_unit: ConsumerUnit = supplier_1.products[SKU3_ID].consumer
        manufacture_unit: ManufactureUnit = supplier_1.products[SKU1_ID].manufacture
        storage_unit: StorageUnit = supplier_1.storage

        consumer_node_index = consumer_unit.data_model_index
        manufacture_node_index = manufacture_unit.data_model_index
        storage_node_index = storage_unit.data_model_index

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

        manufacture_features = (
            "id",
            "facility_id",
            "start_manufacture_quantity",
            "sku_id",
            "in_pipeline_quantity",
            "finished_quantity",
            "product_unit_id",
        )
        storage_features = ("id", "facility_id")

        consumer_nodes = env.snapshot_list["consumer"]
        manufacture_nodes = env.snapshot_list["manufacture"]
        storage_nodes = env.snapshot_list["storage"]

        # ##################################### Before reset #####################################
        action_consumer = ConsumerAction(consumer_unit.id, SKU3_ID, supplier_3.id, 20, "train")
        action_manufacture = ManufactureAction(manufacture_unit.id, 5)

        expect_tick = 100

        # Save the env.metric of each tick into env_metric_1
        env_metric_1: Dict[int, dict] = defaultdict(dict)

        random_tick: List[int] = []

        # The purpose is to randomly perform the order operation
        for i in range(30):
            random_tick.append(random.randint(0, 90))

        # Store the information about the snapshot unit of each tick in states_1
        states_1_consumer: Dict[int, list] = defaultdict(list)
        states_1_manufacture: Dict[int, list] = defaultdict(list)
        states_1_storage: Dict[int, list] = defaultdict(list)

        for i in range(expect_tick):

            if i in random_tick:
                env.step([action_manufacture])
                i += 1
                states_1_manufacture[i] = list(
                    manufacture_nodes[i:manufacture_node_index:manufacture_features]
                    .flatten()
                    .astype(
                        np.int,
                    ),
                )
                env_metric_1[i] = env.metrics
                continue

            env.step([action_consumer])
            env_metric_1[i] = env.metrics
            states_1_consumer[i] = list(
                consumer_nodes[i:consumer_node_index:consumer_features].flatten().astype(np.int),
            )

            states_1_storage[i] = list(storage_nodes[i:storage_node_index:storage_features].flatten().astype(np.int))
            states_1_storage[i].append(
                list(storage_nodes[i:storage_node_index:"product_quantity"].flatten().astype(np.int)),
            )
            states_1_storage[i].append(
                list(storage_nodes[i:storage_node_index:"remaining_space"].flatten().astype(np.int)),
            )

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
        states_2_consumer: Dict[int, list] = defaultdict(list)
        states_2_manufacture: Dict[int, list] = defaultdict(list)
        states_2_storage: Dict[int, list] = defaultdict(list)

        for i in range(expect_tick):

            if i in random_tick:
                env.step([action_manufacture])
                i += 1
                states_2_manufacture[i] = list(
                    manufacture_nodes[i:manufacture_node_index:manufacture_features]
                    .flatten()
                    .astype(
                        np.int,
                    ),
                )
                env_metric_2[i] = env.metrics
                continue

            env.step([action_consumer])
            env_metric_2[i] = env.metrics
            states_2_consumer[i] = list(
                consumer_nodes[i:consumer_node_index:consumer_features].flatten().astype(np.int),
            )

            states_2_storage[i] = list(storage_nodes[i:storage_node_index:storage_features].flatten().astype(np.int))
            states_2_storage[i].append(
                list(
                    storage_nodes[i:storage_node_index:"product_quantity"].flatten().astype(np.int),
                ),
            )
            states_2_storage[i].append(
                list(storage_nodes[i:storage_node_index:"remaining_space"].flatten().astype(np.int)),
            )

        expect_tick = 100
        for i in range(expect_tick):
            self.assertEqual(list(states_1_consumer[i]), list(states_2_consumer[i]))
            self.assertEqual(list(states_1_manufacture[i]), list(states_2_manufacture[i]))
            self.assertEqual(list(states_1_storage[i]), list(states_2_storage[i]))
            self.assertEqual(list(env_metric_1[i].values()), list(env_metric_2[i].values()))


if __name__ == "__main__":
    unittest.main()
