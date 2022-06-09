# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest
import numpy as np

from maro.simulator.scenarios.supply_chain import FacilityBase, ConsumerAction
from maro.simulator.scenarios.supply_chain.business_engine import SupplyChainBusinessEngine
from maro.simulator.scenarios.supply_chain.order import Order

from tests.supply_chain.common import build_env, get_product_dict_from_storage, SKU1_ID, SKU3_ID


class MyTestCase(unittest.TestCase):
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
        consumer_unit = warehouse_1.products[SKU3_ID].consumer

        order_1 = Order(
            src_facility=supplier_3,
            dest_facility=warehouse_1,
            sku_id=SKU3_ID,
            quantity=10,
            vehicle_type="train",
            creation_tick=env.tick,
            expected_finish_tick=env.tick + 7,
        )

        # There are 2 "train" in total, and 1 left after scheduling this order.
        consumer_unit._update_open_orders(warehouse_1.id, SKU3_ID, 10)
        distribution_unit.place_order(order_1)
        distribution_unit.try_schedule_orders(env.tick)
        self.assertEqual(0, len(distribution_unit._order_queues["train"]))
        self.assertEqual(0, sum([order.quantity for order in distribution_unit._order_queues["train"]]))

        # add another order, it would be successfully scheduled, but none available vehicle left now.
        order_2 = Order(
            src_facility=supplier_3,
            dest_facility=warehouse_1,
            sku_id=SKU3_ID,
            quantity=10,
            vehicle_type="train",
            creation_tick=env.tick,
            expected_finish_tick=env.tick + 7,
        )
        consumer_unit._update_open_orders(warehouse_1.id, SKU3_ID, 10)
        distribution_unit.place_order(order_2)
        distribution_unit.try_schedule_orders(env.tick)
        self.assertEqual(0, len(distribution_unit._order_queues["train"]))
        self.assertEqual(0, sum([order.quantity for order in distribution_unit._order_queues["train"]]))

        start_tick = env.tick
        expected_tick = start_tick + 7  # vlt = 7

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
        consumer_unit._update_open_orders(warehouse_1.id, SKU3_ID, 10)
        distribution_unit.place_order(order_3)
        distribution_unit.try_schedule_orders(env.tick)
        self.assertEqual(1, len(distribution_unit._order_queues["train"]))
        self.assertEqual(10, sum([order.required_quantity for order in distribution_unit._order_queues["train"]]))

        while env.tick < expected_tick:
            env.step(None)

        # will arrive at the end of this tick, still on the way.
        assert env.tick == expected_tick
        self.assertEqual(1, len(distribution_unit._order_queues["train"]))
        self.assertEqual(10, sum([order.required_quantity for order in distribution_unit._order_queues["train"]]))

        self.assertEqual(10 * 1, distribution_unit.delay_order_penalty[SKU3_ID])
        self.assertEqual(1 * 10 * 2, distribution_unit.transportation_cost[SKU3_ID])

        env.step(None)

        self.assertEqual(0, len(distribution_unit._order_queues["train"]))
        self.assertEqual(0, sum([order.required_quantity for order in distribution_unit._order_queues["train"]]))

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

        features = ("id", "facility_id", "sku_id", "order_base_cost", "purchased", "received", "order_product_cost")
        IDX_ID, IDX_FACILITY_ID, IDX_SKU_ID, IDX_ORDER_COST = 0, 1, 2, 3
        IDX_PURCHASED, IDX_RECEIVED, IDX_ORDER_PRODUCT_COST = 4, 5, 6

        consumer_nodes = env.snapshot_list["consumer"]

        # ############################## Ask products from Supplier_SKU3 #######################################

        required_quantity = 10
        action = ConsumerAction(sku1_consumer_unit.id, SKU1_ID, supplier_1.id, required_quantity, "train")
        purchase_tick: int = env.tick
        env.step([action])
        env.step(None)
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
        consumer_unit = warehouse_1.products[SKU3_ID].consumer
        warehouse_storage_unit = warehouse_1.storage

        env.step(None)

        start_tick = env.tick
        expected_tick = start_tick + 7
        order = Order(
            src_facility=supplier_3,
            dest_facility=warehouse_1,
            sku_id=SKU3_ID,
            quantity=80,
            vehicle_type="train",
            creation_tick=env.tick,
            expected_finish_tick=env.tick + 7,
        )
        consumer_unit._update_open_orders(warehouse_1.id, SKU3_ID, 80)
        distribution_unit.place_order(order)
        distribution_unit.try_schedule_orders(start_tick)

        while env.tick <= expected_tick:
            # Check the inventory level in target storage
            quantity = get_product_dict_from_storage(
                env, env.frame_index, warehouse_storage_unit.data_model_index
            )[SKU3_ID]

            self.assertEqual(10, quantity)

            # Check the payload in the distribution
            self.assertEqual(1, len(distribution_unit._order_on_the_way[expected_tick]))
            self.assertEqual(warehouse_1, distribution_unit._order_on_the_way[expected_tick][0].dest_facility)
            self.assertEqual(80, distribution_unit._order_on_the_way[expected_tick][0].payload)

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
            self.assertEqual(1, len(distribution_unit._order_on_the_way[expected_tick]))
            self.assertEqual(warehouse_1, distribution_unit._order_on_the_way[expected_tick][0].dest_facility)
            self.assertEqual(80 - 70, distribution_unit._order_on_the_way[expected_tick][0].payload)

            _, _, is_done = env.step(None)


if __name__ == '__main__':
    unittest.main()
