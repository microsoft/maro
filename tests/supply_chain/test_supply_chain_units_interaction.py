# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest
import numpy as np

from maro.simulator.scenarios.supply_chain import (
    ConsumerAction, FacilityBase
)
from maro.simulator.scenarios.supply_chain.business_engine import SupplyChainBusinessEngine

from tests.supply_chain.common import build_env, SKU3_ID


class MyTestCase(unittest.TestCase):
    """
    Units Interaction tests:
        . ConsumerUnit will receive products after vlt (+ 1) days
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

        features = ("id", "facility_id", "sku_id", "order_base_cost", "purchased", "received", "order_product_cost")
        # IDX_ID, IDX_FACILITY_ID, IDX_SKU_ID, IDX_ORDER_COST = 0, 1, 2, 3
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
            self.assertEqual(required_quantity_1, sku3_consumer_unit._open_orders[supplier_3.id])
            env.step(None)

        self.assertEqual(0, sku3_consumer_unit._open_orders[supplier_3.id])

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
            self.assertEqual(required_quantity_2, sku3_consumer_unit._open_orders[supplier_4.id])
            env.step(None)

        self.assertEqual(0, sku3_consumer_unit._open_orders[supplier_4.id])

        expected_frame = env.business_engine.frame_index(expected_tick_2)

        # Not received yet.
        states = consumer_nodes[expected_frame - 1:consumer_node_index:features].flatten().astype(np.int)
        self.assertEqual(0, states[IDX_RECEIVED])

        # received.
        states = consumer_nodes[expected_frame:consumer_node_index:features].flatten().astype(np.int)
        self.assertEqual(required_quantity_2, states[IDX_RECEIVED])


if __name__ == '__main__':
    unittest.main()
