# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest
import numpy as np

from maro.simulator.scenarios.supply_chain import FacilityBase, ConsumerAction
from maro.simulator.scenarios.supply_chain.business_engine import SupplyChainBusinessEngine
from maro.simulator.scenarios.supply_chain.order import Order

from tests.supply_chain.common import build_env, SKU3_ID, FOOD_1_ID


class MyTestCase(unittest.TestCase):
    """
        Consumer test:

        . initial state
        . state after reset
        . set_action directly from code
        . set_action by env.step
        . call on_order_reception directly to simulation order arrived
        . call update_open_orders directly
        . with dynamics sampler
        """

    def test_consumer_init_state(self) -> None:
        """Consumer of sku3 in Supplier_SKU1."""
        env = build_env("case_01", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        supplier_1: FacilityBase = be.world._get_facility_by_name("Supplier_SKU1")
        sku3_consumer_unit = supplier_1.products[SKU3_ID].consumer

        consumer_node_index = sku3_consumer_unit.data_model_index

        features = ("id", "facility_id", "sku_id", "order_base_cost", "purchased", "received", "order_product_cost")
        IDX_ID, IDX_FACILITY_ID, IDX_SKU_ID, IDX_ORDER_COST = 0, 1, 2, 3
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
        self.assertEqual(SKU3_ID, states[IDX_SKU_ID])
        self.assertEqual(0, states[IDX_ORDER_COST])

        env.reset()
        env.step(None)

        states = consumer_nodes[env.frame_index:consumer_node_index:features].flatten().astype(np.int)

        # Nothing happened at tick 0, so most states will be 0
        self.assertEqual(0, states[IDX_PURCHASED])
        self.assertEqual(0, states[IDX_RECEIVED])
        self.assertEqual(0, states[IDX_ORDER_PRODUCT_COST])

        self.assertEqual(sku3_consumer_unit.id, states[IDX_ID])
        self.assertEqual(SKU3_ID, states[IDX_SKU_ID])

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

        features = ("id", "facility_id", "sku_id", "order_base_cost", "purchased", "received", "order_product_cost")
        IDX_ID, IDX_FACILITY_ID, IDX_SKU_ID, IDX_ORDER_COST = 0, 1, 2, 3
        IDX_PURCHASED, IDX_RECEIVED, IDX_ORDER_PRODUCT_COST = 4, 5, 6

        consumer_nodes = env.snapshot_list["consumer"]

        # ############################### Test Action with 0 quantity ######################################
        # zero quantity will be ignored
        action_with_zero = ConsumerAction(sku3_consumer_unit.id, SKU3_ID, supplier_3.id, 0, "train")
        env.step([action_with_zero])

        states = consumer_nodes[env.frame_index:consumer_node_index:features].flatten().astype(np.int)

        # Nothing happened at tick 0, at the action will be recorded
        self.assertEqual(action_with_zero.sku_id, states[IDX_SKU_ID])
        self.assertEqual(action_with_zero.quantity, states[IDX_PURCHASED])

        self.assertEqual(sku3_consumer_unit.id, states[IDX_ID])
        self.assertEqual(SKU3_ID, states[IDX_SKU_ID])

        # ############################### Test Action with positive quantity ######################################
        action = ConsumerAction(sku3_consumer_unit.id, SKU3_ID, supplier_3.id, 1, "train")
        purchased_tick = env.tick
        env.step([action])

        purchased_frame = env.business_engine.frame_index(purchased_tick)
        states = consumer_nodes[purchased_frame:consumer_node_index:features].flatten().astype(np.int)
        self.assertEqual(action.quantity, states[IDX_PURCHASED])
        self.assertEqual(0, states[IDX_RECEIVED])
        self.assertEqual(action.sku_id, states[IDX_SKU_ID])

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
        order = Order(
            src_facility=supplier_3,
            dest_facility=supplier_1,
            sku_id=SKU3_ID,
            quantity=required_quantity,
            vehicle_type="train",
            creation_tick=env.tick,
            expected_finish_tick=None,
        )

        env.step([action])

        # simulate purchased product is arrived by vehicle unit
        sku3_consumer_unit.on_order_reception(order=order, received_quantity=required_quantity, tick=env.tick)

        # now all order is done
        self.assertEqual(0, sku3_consumer_unit._open_orders[supplier_3.id])
        self.assertEqual(required_quantity, sku3_consumer_unit._received)

        # NOTE: we cannot test the received state by calling on_order_reception directly,
        # as it will be cleared by env.step, do it on vehicle unit test.

        env.step(None)

    def test_consumer_unit_dynamics_sampler(self):
        """Tested the store_001  Interaction between consumer unit and dynamics csv data.
           The data file of this test is test_case_ 04.csv"""
        env = build_env("case_04", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)
        env.step(None)
        Store_001: FacilityBase = be.world._get_facility_by_name("Store_001")
        FOOD_1_consumer_unit = Store_001.products[FOOD_1_ID].consumer

        consumer_node_index = FOOD_1_consumer_unit.data_model_index

        features = ("id", "facility_id", "sku_id", "order_base_cost", "purchased", "received", "order_product_cost")
        IDX_ID, IDX_FACILITY_ID, IDX_SKU_ID, IDX_ORDER_COST = 0, 1, 2, 3
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
        self.assertEqual(FOOD_1_ID, states[IDX_SKU_ID])
        self.assertEqual(0, states[IDX_ORDER_COST])

        env.reset()
        env.step(None)

        states = consumer_nodes[env.frame_index:consumer_node_index:features].flatten().astype(np.int)

        # Nothing happened at tick 0, so most states will be 0
        self.assertEqual(0, states[IDX_PURCHASED])
        self.assertEqual(0, states[IDX_RECEIVED])
        self.assertEqual(0, states[IDX_ORDER_PRODUCT_COST])

        self.assertEqual(FOOD_1_consumer_unit.id, states[IDX_ID])
        self.assertEqual(FOOD_1_ID, states[IDX_SKU_ID])

        """test_consumer_action"""

        features = ("id", "facility_id", "sku_id", "order_base_cost", "purchased", "received", "order_product_cost")
        IDX_ID, IDX_FACILITY_ID, IDX_SKU_ID, IDX_ORDER_COST = 0, 1, 2, 3
        IDX_PURCHASED, IDX_RECEIVED, IDX_ORDER_PRODUCT_COST = 4, 5, 6

        consumer_nodes = env.snapshot_list["consumer"]

        # ############################### Test Action with 0 quantity ######################################
        # zero quantity will be ignore
        action_with_zero = ConsumerAction(FOOD_1_consumer_unit.id, FOOD_1_ID, Store_001.id, 0, "train")
        env.step([action_with_zero])

        states = consumer_nodes[env.frame_index:consumer_node_index:features].flatten().astype(np.int)

        # Nothing happened at tick 0, at the action will be recorded
        self.assertEqual(action_with_zero.sku_id, states[IDX_SKU_ID])
        self.assertEqual(action_with_zero.quantity, states[IDX_PURCHASED])

        self.assertEqual(FOOD_1_consumer_unit.id, states[IDX_ID])
        self.assertEqual(FOOD_1_ID, states[IDX_SKU_ID])

        # ############################### Test Action with positive quantity ######################################
        action = ConsumerAction(FOOD_1_consumer_unit.id, FOOD_1_ID, Store_001.id, 0, "train")
        env.step([action])

        self.assertEqual(action.quantity, FOOD_1_consumer_unit._purchased)
        self.assertEqual(0, FOOD_1_consumer_unit._received)

        states = consumer_nodes[env.frame_index:consumer_node_index:features].flatten().astype(np.int)

        # action field should be recorded
        self.assertEqual(action.sku_id, states[IDX_SKU_ID])

        self.assertEqual(action.quantity, states[IDX_PURCHASED])

        # no receives
        self.assertEqual(0, states[IDX_RECEIVED])


if __name__ == '__main__':
    unittest.main()
