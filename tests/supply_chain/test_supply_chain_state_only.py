import os
import unittest
import numpy as np

from maro.simulator import Env
from maro.simulator.scenarios.supply_chain import FacilityBase, ConsumerAction, StorageUnit
from maro.simulator.scenarios.supply_chain.business_engine import SupplyChainBusinessEngine
from maro.simulator.scenarios.supply_chain.order import Order
from maro.simulator.scenarios.supply_chain.sku_dynamics_sampler import OneTimeSkuPriceDemandSampler, \
    DataFileDemandSampler


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
        Consumer test:

        . initial state
        . state after reset
        . set_action directly from code
        . set_action by env.step
        . call on_order_reception directly to simulation order arrived
        . call update_open_orders directly
        . with dynamics sampler
        """

    def test_consumer_state_only(self) -> None:
        """Test the 'pending_order_daily' of the consumer unit."""
        env = build_env("case_05", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        supplier_3 = be.world._get_facility_by_name("Supplier_SKU3")
        distribution_unit = supplier_3.distribution
        warehouse_1 = be.world._get_facility_by_name("Warehouse_001")

        distribution_unit = supplier_3.distribution

        order = Order(warehouse_1, SKU3_ID, 10, "train")

        # There are 2 "train" in total, and 1 left after scheduling this order.
        distribution_unit.place_order(env.tick, order)
        self.assertEqual(0, len(distribution_unit._order_queues["train"]))
        self.assertEqual(0, sum([order.quantity for order in distribution_unit._order_queues["train"]]))
        warehouse_1_sku3 = 11

        # Here the vlt of "train" is less than "pending_order_daily" length
        self.assertEqual([0, 0, 0, 10], env.metrics['products'][warehouse_1_sku3]['pending_order_daily'])

        # add another order, it would be successfully scheduled, but none available vehicle left now.
        distribution_unit.place_order(env.tick, order)
        self.assertEqual(0, len(distribution_unit._order_queues["train"]))
        self.assertEqual(0, sum([order.quantity for order in distribution_unit._order_queues["train"]]))
        self.assertEqual([0, 0, 0, 10 + 10], env.metrics['products'][warehouse_1_sku3]['pending_order_daily'])

        start_tick = env.tick
        expected_tick = start_tick + 3  # vlt = 3

        # 3rd order, will cause the pending order increase
        distribution_unit.place_order(env.tick, order)
        self.assertEqual(1, len(distribution_unit._order_queues["train"]))
        self.assertEqual(10, sum([order.quantity for order in distribution_unit._order_queues["train"]]))

        # while env.tick < expected_tick:
        #     env.step(None)
        env.step(None)
        self.assertEqual(20, env.metrics['products'][warehouse_1_sku3]['pending_order_daily'][2])
        env.step(None)
        self.assertEqual(20, env.metrics['products'][warehouse_1_sku3]['pending_order_daily'][1])
        env.step(None)
        self.assertEqual(20, env.metrics['products'][warehouse_1_sku3]['pending_order_daily'][0])
        env.step(None)
        self.assertEqual(0, env.metrics['products'][warehouse_1_sku3]['pending_order_daily'][0])
        # will arrive at the end of this tick, still on the way.
        assert env.tick == expected_tick
        self.assertEqual(1, len(distribution_unit._order_queues["train"]))
        self.assertEqual(10, sum([order.quantity for order in distribution_unit._order_queues["train"]]))
        # self.assertIs([0, 0, 0, 0], env.metrics['products'][11]['pending_order_daily'])

        self.assertEqual(10 * 1, distribution_unit.delay_order_penalty[SKU3_ID])
        self.assertEqual(1 * 10 * 2, distribution_unit.transportation_cost[SKU3_ID])

        env.step(None)

        self.assertEqual(0, len(distribution_unit._order_queues["train"]))
        self.assertEqual(0, sum([order.quantity for order in distribution_unit._order_queues["train"]]))

        self.assertEqual(0, distribution_unit.delay_order_penalty[SKU3_ID])
        self.assertEqual(1 * 10 * 1, distribution_unit.transportation_cost[SKU3_ID])
        self.assertEqual(10, env.metrics['products'][warehouse_1_sku3]['pending_order_daily'][2])

        self.assertEqual(0, env.metrics['products'][warehouse_1_sku3]['pending_order_daily'][3])
        distribution_unit.place_order(env.tick, order)
        self.assertEqual(10, env.metrics['products'][warehouse_1_sku3]['pending_order_daily'][3])

        start_tick = env.tick
        expected_tick = start_tick + 3 - 1  # vlt = 3
        while env.tick < expected_tick:
            env.step(None)

        self.assertEqual(10, env.metrics['products'][warehouse_1_sku3]['pending_order_daily'][0])

    def test_seller_state_only(self) -> None:
        """Test "sale_mean" and "sale_std"""

        env = build_env("case_05", 600)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)
        Store_001: FacilityBase = be.world._get_facility_by_name("Store_001")
        storeproductunit_sku1, storeproductunit_sku2, storeproductunit_sku3 = 1, 3, 2
        self.assertEqual([1, 1, 1, 1, 1, 1], Store_001.children[storeproductunit_sku1].seller._sale_hist)
        self.assertEqual([2, 2, 2, 2, 2, 2], Store_001.children[storeproductunit_sku2].seller._sale_hist)
        self.assertEqual([3, 3, 3, 3, 3, 3], Store_001.children[storeproductunit_sku3].seller._sale_hist)

        env.step(None)
        # The demand in the data file should be added after env.step, and now it is filled with 0 if it is not implemented.
        self.assertEqual([1, 1, 1, 1, 1, 0], Store_001.children[storeproductunit_sku1].seller._sale_hist)
        self.assertEqual([2, 2, 2, 2, 2, 0], Store_001.children[storeproductunit_sku2].seller._sale_hist)
        self.assertEqual([3, 3, 3, 3, 3, 0], Store_001.children[storeproductunit_sku3].seller._sale_hist)

        # The result should be (1+1+1+1+1)/6=0.8333333333333334
        self.assertEqual(0.8333333333333334, env.metrics['products'][26]['sale_mean'])
        # The result should be (3+3+3+3+3)/6=2.5
        self.assertEqual(2.5, env.metrics['products'][29]['sale_mean'])
        # The result should be (2+2+2+2+2)/6=1.6666666666666667
        self.assertEqual(1.6666666666666667, env.metrics['products'][29]['sale_mean'])

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
        # zero quantity will be ignored
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
        consumer_nodes = env.snapshot_list["consumer"]
        consumer_node_index = sku3_consumer_unit.data_model_index

        features = (
            "id", "facility_id", "product_id", "order_base_cost", "purchased", "received", "order_product_cost", "id")
        IDX_ID, IDX_FACILITY_ID, IDX_PRODUCT_ID, IDX_ORDER_COST = 0, 1, 2, 3
        IDX_PURCHASED, IDX_RECEIVED, IDX_ORDER_PRODUCT_COST = 4, 5, 6
        states = consumer_nodes[env.frame_index:consumer_node_index:features].flatten().astype(np.int)

        required_quantity = 550
        action = ConsumerAction(sku3_consumer_unit.id, SKU3_ID, supplier_3.id, required_quantity, "train")

        env.step([action])
        states = consumer_nodes[env.frame_index:consumer_node_index:features].flatten().astype(np.int)

        # simulate purchased product is arrived by vehicle unit
        sku3_consumer_unit.on_order_reception(supplier_3.id, SKU3_ID, required_quantity, required_quantity)

        # now all order is done
        self.assertEqual(0, sku3_consumer_unit._open_orders[supplier_3.id][SKU3_ID])
        self.assertEqual(required_quantity, sku3_consumer_unit._received)

        # NOTE: we cannot test the received state by calling on_order_reception directly,
        # as it will be cleared by env.step, do it on vehicle unit test.

        env.step(None)

    def test_consumer_unit_dynamics_sampler(self):
        """Tested the store_001  Interaction between consumer unit and dynamics csv data.
           The data file of this test is test_case_ 04.csv"""
        env = build_env("case_04", 600)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)
        env.step(None)
        Store_001: FacilityBase = be.world._get_facility_by_name("Store_001")
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
        action_with_zero = ConsumerAction(FOOD_1_consumer_unit.id, FOOD_1_ID, Store_001.id, 0, "train")
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
