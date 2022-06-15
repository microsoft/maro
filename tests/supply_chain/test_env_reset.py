# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license
import random
import unittest
from collections import defaultdict
from typing import Dict, List

import numpy as np

from maro.simulator.scenarios.supply_chain import FacilityBase, ConsumerAction, ManufactureAction, StorageUnit
from maro.simulator.scenarios.supply_chain.business_engine import SupplyChainBusinessEngine
from maro.simulator.scenarios.supply_chain.order import Order

from tests.supply_chain.common import build_env, SKU3_ID, FOOD_1_ID, get_product_dict_from_storage


class MyTestCase(unittest.TestCase):
    """
        . consumer unit test
        . distribution unit test
        . manufacture unit test
        . seller unit test
        . storage unit test
        """

    def test_consumer_unit_reset(self) -> None:
        """Test whether reset updates the consumer unit completely"""
        env = build_env("case_01", 500)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        env.step(None)

        supplier_1: FacilityBase = be.world._get_facility_by_name("Supplier_SKU1")
        supplier_3: FacilityBase = be.world._get_facility_by_name("Supplier_SKU3")
        sku3_consumer_unit = supplier_1.products[SKU3_ID].consumer

        consumer_node_index = sku3_consumer_unit.data_model_index

        features = ("id", "facility_id", "sku_id", "order_base_cost", "purchased", "received", "order_product_cost",
                    "latest_consumptions", "in_transit_quantity")

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

    def test_distribution_unit_reset(self) -> None:
        """Test initial state of the DistributionUnit of Supplier_SKU3.Test distribution unit reset"""
        env = build_env("case_02", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        env.step(None)

        supplier_3 = be.world._get_facility_by_name("Supplier_SKU3")
        warehouse_1 = be.world._get_facility_by_name("Warehouse_001")

        distribution_unit = supplier_3.distribution
        distribution_node_index = distribution_unit.data_model_index
        distribution_nodes = env.snapshot_list["distribution"]

        features = ("id", "facility_id", "pending_order_number", "pending_product_quantity")

        # ##################################### Before reset #####################################

        order_1 = Order(src_facility=supplier_3,
                        dest_facility=warehouse_1,
                        sku_id=SKU3_ID,
                        quantity=10,
                        vehicle_type="train",
                        creation_tick=env.tick,
                        expected_finish_tick=env.tick + 7, )

        # There are 2 "train" in total, and 1 left after scheduling this order.
        distribution_unit.place_order(order_1)
        distribution_unit.try_schedule_orders(env.tick)
        self.assertEqual(0, len(distribution_unit._order_queues["train"]))
        self.assertEqual(0, sum([order.quantity for order in distribution_unit._order_queues["train"]]))

        order_2 = Order(
            src_facility=supplier_3,
            dest_facility=warehouse_1,
            sku_id=SKU3_ID,
            quantity=10,
            vehicle_type="train",
            creation_tick=env.tick,
            expected_finish_tick=env.tick + 7,
        )

        distribution_unit.place_order(order_2)
        distribution_unit.try_schedule_orders(env.tick)
        self.assertEqual(0, len(distribution_unit._order_queues["train"]))
        self.assertEqual(0, sum([order.quantity for order in distribution_unit._order_queues["train"]]))

        # 3rd order, will cause the pending order increase
        order_3 = Order(
            src_facility=supplier_3,
            dest_facility=warehouse_1,
            sku_id=SKU3_ID,
            quantity=10,
            vehicle_type="train",
            creation_tick=env.tick,
            expected_finish_tick=env.tick + 7,
        )
        distribution_unit.place_order(order_3)
        distribution_unit.try_schedule_orders(env.tick)
        self.assertEqual(1, len(distribution_unit._order_queues["train"]))
        self.assertEqual(10, sum([order.required_quantity for order in distribution_unit._order_queues["train"]]))

        env.step(None)

        # The purpose is to randomly perform the order operation
        random_tick: List[int] = []
        for j in range(10):
            random_tick.append(random.randint(5, 100))

        expect_tick = 100

        # Save the env.metric of each tick into env_metric_1
        env_metric_1: Dict[int, dict] = defaultdict(dict)

        # Store the information about the snapshot distribution unit of each tick in states_1
        states_1: Dict[int, list] = defaultdict(list)

        for i in range(expect_tick):
            if i in random_tick:
                order = Order(
                    src_facility=supplier_3,
                    dest_facility=warehouse_1,
                    sku_id=SKU3_ID,
                    quantity=10,
                    vehicle_type="train",
                    creation_tick=env.tick,
                    expected_finish_tick=env.tick + 7,
                )
                distribution_unit.place_order(order)
                distribution_unit.try_schedule_orders(env.tick)
            env.step(None)
            env_metric_1[i] = env.metrics
            states_1[i] = distribution_nodes[i:distribution_node_index:features].flatten().astype(np.int)

        # ####################### Test whether reset updates the distribution unit completely ################
        env.reset()
        env.step(None)

        distribution_nodes = env.snapshot_list["distribution"]

        # snapshot should reset after env.reset().
        states = distribution_nodes[1:distribution_node_index:features].flatten().astype(np.int)
        self.assertEqual([0, 0, 0, 0], list(states))

        # Do the same as before env.reset().
        distribution_unit.place_order(order_1)
        distribution_unit.try_schedule_orders(env.tick)

        distribution_unit.place_order(order_2)
        distribution_unit.try_schedule_orders(env.tick)
        self.assertEqual(0, len(distribution_unit._order_queues["train"]))
        self.assertEqual(0, sum([order.quantity for order in distribution_unit._order_queues["train"]]))

        distribution_unit.place_order(order_3)
        distribution_unit.try_schedule_orders(env.tick)
        self.assertEqual(1, len(distribution_unit._order_queues["train"]))
        self.assertEqual(10, sum([order.required_quantity for order in distribution_unit._order_queues["train"]]))

        env.step(None)

        expect_tick = 100
        # Save the env.metric of each tick into env_metric_2.
        env_metric_2: Dict[int, dict] = defaultdict(dict)

        # Store the information about the snapshot distribution unit of each tick in states_2.
        states_2: Dict[int, list] = defaultdict(list)

        for i in range(expect_tick):
            if i in random_tick:
                order = Order(
                    src_facility=supplier_3,
                    dest_facility=warehouse_1,
                    sku_id=SKU3_ID,
                    quantity=10,
                    vehicle_type="train",
                    creation_tick=env.tick,
                    expected_finish_tick=env.tick + 7,
                )
                distribution_unit.place_order(order)
                distribution_unit.try_schedule_orders(env.tick)
            env.step(None)
            env_metric_2[i] = env.metrics
            states_2[i] = distribution_nodes[i:distribution_node_index:features].flatten().astype(np.int)

        expect_tick = 100
        for i in range(expect_tick):
            self.assertEqual(list(states_1[i]), list(states_2[i]))
            self.assertEqual(list(env_metric_1[i].values()), list(env_metric_2[i].values()))

    def test_manufacture_unit_reset(self) -> None:
        """Test sku3 manufacturing. -- Supplier_SKU3.Test manufacture unit reset"""
        env = build_env("case_01", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        supplier_3: FacilityBase = be.world._get_facility_by_name("Supplier_SKU3")
        sku3_storage_index = supplier_3.storage.data_model_index
        manufacture_sku3_unit = supplier_3.products[SKU3_ID].manufacture
        sku3_manufacture_index = manufacture_sku3_unit.data_model_index

        storage_nodes = env.snapshot_list["storage"]

        manufacture_features = (
            "id", "facility_id", "start_manufacture_quantity", "sku_id", "in_pipeline_quantity", "finished_quantity",
            "product_unit_id",
        )

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
        action = ManufactureAction(manufacture_sku3_unit.id, 1)

        expect_tick = 30
        env_metric_1: Dict[int, dict] = defaultdict(dict)
        states_1: Dict[int, list] = defaultdict(list)
        random_tick: List[int] = []
        manufacture_nodes = env.snapshot_list["manufacture"]

        for i in range(10):
            random_tick.append(random.randint(1, 30))

        for i in range(expect_tick):
            env.step([action])
            if i in random_tick:
                env.step([ManufactureAction(manufacture_sku3_unit.id, 0)])
            env_metric_1[i] = env.metrics
            states_1[i] = manufacture_nodes[i:sku3_manufacture_index:manufacture_features].flatten().astype(np.int)

        # ############################### Test whether reset updates the distribution unit completely ################
        env.reset()
        env.step(None)

        states = manufacture_nodes[1:sku3_manufacture_index:manufacture_features].flatten().astype(np.int)
        self.assertEqual([0, 0, 0, 0, 0, 0, 0], list(states))

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

        expect_tick = 30
        env_metric_2: Dict[int, dict] = defaultdict(dict)
        states_2: Dict[int, list] = defaultdict(list)
        manufacture_nodes = env.snapshot_list["manufacture"]

        for i in range(expect_tick):
            env.step([action])
            if i in random_tick:
                env.step([ManufactureAction(manufacture_sku3_unit.id, 0)])
            env_metric_2[i] = env.metrics
            states_2[i] = manufacture_nodes[i:sku3_manufacture_index:manufacture_features].flatten().astype(np.int)

        expect_tick = 30
        for i in range(expect_tick):
            self.assertEqual(list(states_1[i]), list(states_2[i]))
            self.assertEqual(list(env_metric_1[i].values()), list(env_metric_2[i].values()))

    def test_seller_unit_dynamics_sampler(self):
        """Tested the store_001  Interaction between seller unit and dynamics csv data.
           The data file of this test is test_case_ 04.csv"""
        env = build_env("case_04", 600)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        env.step(None)
        Store_001: FacilityBase = be.world._get_facility_by_name("Store_001")
        seller_unit = Store_001.products[FOOD_1_ID].seller

        seller_node_index = seller_unit.data_model_index

        seller_nodes = env.snapshot_list["seller"]

        features = ("sold", "demand", "total_sold", "total_demand", "backlog_ratio", "facility_id", "product_unit_id",)

        self.assertEqual(20, seller_unit.sku_id)

        # NOTE: this simple seller unit return demands that same as current tick

        # Tick 0 will have demand == 10.first row of data after preprocessing data.
        # from sample_preprocessed.csv
        self.assertEqual(10, seller_unit._sold)
        self.assertEqual(10, seller_unit._demand)
        self.assertEqual(10, seller_unit._total_sold)

        expect_tick = 12
        env_metric_1: Dict[int, dict] = defaultdict(dict)
        states_1: Dict[int, list] = defaultdict(list)
        for i in range(expect_tick):
            env.step(None)
            env_metric_1[i] = env.metrics
            states_1[i] = seller_nodes[i:seller_node_index:features].flatten().astype(np.int)

        # ############################### Test whether reset updates the distribution unit completely ################
        env.reset()
        env.step(None)
        states = seller_nodes[1:seller_node_index:features].flatten().astype(np.int)
        self.assertEqual([0, 0, 0, 0, 0, 0, 0], list(states))

        expect_tick = 12

        env_metric_2: Dict[int, dict] = defaultdict(dict)
        states_2: Dict[int, list] = defaultdict(list)
        for i in range(expect_tick):
            env.step(None)
            env_metric_2[i] = env.metrics
            states_2[i] = seller_nodes[i:seller_node_index:features].flatten().astype(np.int)

        for i in range(expect_tick):
            self.assertEqual(list(states_1[i]), list(states_2[i]))
            self.assertEqual(list(env_metric_1[i].values()), list(env_metric_2[i].values()))

    def test_storage_unit_reset(self) -> None:
        """Facility with single SKU. -- Supplier_SKU3"""
        env = build_env("case_01", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        env.step(None)

        supplier_3: FacilityBase = be.world._get_facility_by_name("Supplier_SKU3")
        storage_unit: StorageUnit = supplier_3.storage
        storage_node_index = storage_unit.data_model_index

        storage_nodes = env.snapshot_list["storage"]

        features = ("id", "facility_id",)

        # ############################### Take more than existing ######################################

        # which this setting, it will return false, as no enough product for ous

        expect_tick = 10
        env_metric_1: Dict[int, dict] = defaultdict(dict)
        states_1: Dict[int, list] = defaultdict(list)
        for i in range(expect_tick):
            env.step(None)
            env_metric_1[i] = env.metrics
            states_1[i] = list(storage_nodes[i:storage_node_index:features].flatten().astype(np.int))
            states_1[i].append(storage_nodes[i:storage_node_index:"product_id_list"].flatten().astype(np.int).sum())
            states_1[i].append(storage_nodes[i:storage_node_index:"product_quantity"].flatten().astype(np.int).sum())
            states_1[i].append(storage_nodes[i:storage_node_index:"remaining_space"].flatten().astype(np.int).sum())

        # ############################### Test whether reset updates the distribution unit completely ################
        env.reset()
        env.step(None)

        states = storage_nodes[1:storage_node_index:features].flatten().astype(np.int)
        self.assertEqual([0, 0], list(states))

        expect_tick = 10

        env_metric_2: Dict[int, dict] = defaultdict(dict)
        states_2: Dict[int, list] = defaultdict(list)
        for i in range(expect_tick):
            env.step(None)
            env_metric_2[i] = env.metrics
            states_2[i] = list(storage_nodes[i:storage_node_index:features].flatten().astype(np.int))
            states_2[i].append(storage_nodes[i:storage_node_index:"product_id_list"].flatten().astype(np.int).sum())
            states_2[i].append(storage_nodes[i:storage_node_index:"product_quantity"].flatten().astype(np.int).sum())
            states_2[i].append(storage_nodes[i:storage_node_index:"remaining_space"].flatten().astype(np.int).sum())

        for i in range(expect_tick):
            self.assertEqual(list(states_1[i]), list(states_2[i]))
            self.assertEqual(list(env_metric_1[i].values()), list(env_metric_2[i].values()))


if __name__ == '__main__':
    unittest.main()
