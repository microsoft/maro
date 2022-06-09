import os
import unittest
import numpy as np

from maro.simulator import Env
from maro.simulator.scenarios.supply_chain import FacilityBase, ProductUnit
from maro.simulator.scenarios.supply_chain.business_engine import SupplyChainBusinessEngine
from maro.simulator.scenarios.supply_chain.order import Order


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
        state only  test:
        . seller_state_only
            . "sale_mean"
            . "sale_hist"
        . distribution_state_only
            . "pending_order"
            . "in_transit_orders"
            . "pending_order_daily"
    """

    def test_distribution_state_only_small_vlt(self) -> None:
        """Test the "pending_order_daily" of the distribution unit when vlt is less than "pending_order_daily" length.
        """
        env = build_env("case_05", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        supplier_3 = be.world._get_facility_by_name("Supplier_SKU3")
        warehouse_1 = be.world._get_facility_by_name("Warehouse_001")
        distribution_unit = supplier_3.distribution

        consumer_unit = warehouse_1.products[SKU3_ID].consumer
        env.step(None)

        #  vlt is greater than len(pending_order_len), which will cause the pending order to increase
        order_1 = Order(supplier_3, warehouse_1, SKU3_ID, 1, "train", env.tick, None)
        distribution_unit.place_order(order_1)
        consumer_unit._update_open_orders(warehouse_1.id, SKU3_ID, 1)
        self.assertEqual(1, len(distribution_unit._order_queues["train"]))
        self.assertEqual(1, sum([order.required_quantity for order in distribution_unit._order_queues["train"]]))
        self.assertEqual(0 * 1, distribution_unit.transportation_cost[SKU3_ID])

        env.step(None)
        # Here the vlt of "train" is less than "pending_order_daily" length
        self.assertEqual(
            [0, 0, 1, 0], list(env.metrics["products"][warehouse_1.products[SKU3_ID].id]["pending_order_daily"]),
        )
        self.assertEqual(1 * 1, distribution_unit.transportation_cost[SKU3_ID])

        # add another order, it would be successfully scheduled.
        order_2 = Order(supplier_3, warehouse_1, SKU3_ID, 2, "train", env.tick, None)
        distribution_unit.place_order(order_2)
        consumer_unit._update_open_orders(warehouse_1.id, SKU3_ID, 2)
        self.assertEqual(1, len(distribution_unit._order_queues["train"]))
        self.assertEqual(2, sum([order.required_quantity for order in distribution_unit._order_queues["train"]]))

        self.assertEqual(
            [0, 0,  1, 0], list(env.metrics["products"][warehouse_1.products[SKU3_ID].id]["pending_order_daily"]),
        )

        start_tick = env.tick
        expected_tick = start_tick + 3

        # vlt is greater than len(pending_order_len), which will cause the pending order to increase.
        # Add another order, which will be successfully arranged, but there are no extra vehicles now.
        order_3 = Order(supplier_3, warehouse_1, SKU3_ID, 3, "train", env.tick, None)
        distribution_unit.place_order(order_3)
        consumer_unit._update_open_orders(warehouse_1.id, SKU3_ID, 3)

        self.assertEqual(2, len(distribution_unit._order_queues["train"]))
        self.assertEqual(5, sum([order.required_quantity for order in distribution_unit._order_queues["train"]]))
        # For the third order, there are two trains in total, sqo the third order will not enter pending_order_daily
        # after the step.

        env.step(None)
        self.assertEqual(
            [0, 1, 2, 0], list(env.metrics["products"][warehouse_1.products[SKU3_ID].id]["pending_order_daily"]),
        )
        self.assertEqual(1 * (1+2), distribution_unit.transportation_cost[SKU3_ID])

        env.step(None)
        self.assertEqual(
            [1, 2, 0, 0], list(env.metrics["products"][warehouse_1.products[SKU3_ID].id]["pending_order_daily"]),
        )

        self.assertEqual(1, len(distribution_unit._order_queues["train"]))
        self.assertEqual(3, sum([order.required_quantity for order in distribution_unit._order_queues["train"]]))

        self.assertEqual(1 * (1+2), distribution_unit.transportation_cost[SKU3_ID])

        env.step(None)
        self.assertEqual(
            [2, 0, 0, 3], list(env.metrics["products"][warehouse_1.products[SKU3_ID].id]["pending_order_daily"]),
        )

        # will arrive at the end of this tick, still on the way.
        self.assertEqual(0, len(distribution_unit._order_queues["train"]))
        self.assertEqual(0, sum([order.required_quantity for order in distribution_unit._order_queues["train"]]))

        self.assertEqual(1 * (2+3), distribution_unit.transportation_cost[SKU3_ID])
        self.assertEqual(10 * 0, distribution_unit.delay_order_penalty[SKU3_ID])

        assert env.tick == expected_tick
        env.step(None)
        self.assertEqual(
            [0, 0, 3, 0], list(env.metrics["products"][warehouse_1.products[SKU3_ID].id]["pending_order_daily"]),
        )

        self.assertEqual(0, len(distribution_unit._order_queues["train"]))
        self.assertEqual(0, sum([order.required_quantity for order in distribution_unit._order_queues["train"]]))

        self.assertEqual(0, distribution_unit.delay_order_penalty[SKU3_ID])
        self.assertEqual(1 * 3, distribution_unit.transportation_cost[SKU3_ID])

        env.step(None)

        consumer_unit._update_open_orders(warehouse_1.id, SKU3_ID, 1)
        distribution_unit.place_order(order_1)

        self.assertEqual(
            [0, 3, 0, 0], list(env.metrics["products"][warehouse_1.products[SKU3_ID].id]["pending_order_daily"]),
        )

        self.assertEqual(1, len(distribution_unit._order_queues["train"]))
        self.assertEqual(1, sum([order.required_quantity for order in distribution_unit._order_queues["train"]]))

        self.assertEqual(0, distribution_unit.delay_order_penalty[SKU3_ID])
        self.assertEqual(1 * 3, distribution_unit.transportation_cost[SKU3_ID])

        env.step(None)

        self.assertEqual(1 * 3 + 1 * 1, distribution_unit.transportation_cost[SKU3_ID])

        self.assertEqual(
            [3, 0, 1, 0], list(env.metrics["products"][warehouse_1.products[SKU3_ID].id]["pending_order_daily"]),
        )

        start_tick = env.tick
        expected_tick = start_tick + 3 - 1  # vlt = 3
        while env.tick < expected_tick:
            env.step(None)

        self.assertEqual(
            [1, 0, 0, 0], list(env.metrics["products"][warehouse_1.products[SKU3_ID].id]["pending_order_daily"]),
        )
        self.assertEqual(1 * 1 * 1, distribution_unit.transportation_cost[SKU3_ID])

    def test_distribution_state_only_bigger_vlt(self) -> None:
        """Tests the "pending_order_daily" of the distribution unit when vlt is greater than the
        "pending_order_daily" length. """

        env = build_env("case_05", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        warehouse_1 = be.world._get_facility_by_name("Warehouse_001")
        retailer_1: FacilityBase = be.world._get_facility_by_name("Retailer_001")
        warehouse_1_id, retailer_1_id = 6, 13

        warehouse_1_distribution_unit = warehouse_1.distribution
        self.assertEqual(0, len(warehouse_1_distribution_unit._order_queues["train"]))

        env.step(None)

        consumer_unit = retailer_1.products[SKU2_ID].consumer

        order_1 = Order(warehouse_1, retailer_1, SKU2_ID, 1, "train", env.tick, None)
        warehouse_1_distribution_unit.place_order(order_1)
        consumer_unit._update_open_orders(retailer_1.id, SKU2_ID, 1)
        # The vlt configuration of this topology is 5.
        self.assertEqual(1, len(warehouse_1_distribution_unit._order_queues["train"]))

        # After env.step runs, where tick is 1. order_1 will arrive at tick=5. order_2 will arrive at tick=6.
        env.step(None)

        self.assertEqual(0, len(warehouse_1_distribution_unit._order_queues["train"]))
        self.assertEqual(
            [0, 0, 0, 0], list(env.metrics["products"][retailer_1.products[SKU2_ID].id]["pending_order_daily"]),
        )

        order_2 = Order(warehouse_1, retailer_1, SKU2_ID, 2, "train", env.tick, None)
        warehouse_1_distribution_unit.place_order(order_2)
        consumer_unit._update_open_orders(retailer_1.id, SKU2_ID, 2)

        # The vlt configuration of this topology is 5.
        self.assertEqual(1, len(warehouse_1_distribution_unit._order_queues["train"]))

        # After env.step runs, where tick is 2. order_1 will arrive at tick=5. order_2 will arrive at tick=6.
        env.step(None)

        self.assertEqual(
            [0, 0, 0, 1], list(env.metrics["products"][retailer_1.products[SKU2_ID].id]["pending_order_daily"]),
        )

        # After env.step runs, where tick is 3. order_1 will arrive at tick=5. order_2 will arrive at tick=6.
        env.step(None)

        self.assertEqual(
            [0, 0, 1, 2], list(env.metrics["products"][retailer_1.products[SKU2_ID].id]["pending_order_daily"]),
        )

        self.assertEqual(3, env.metrics["facilities"][retailer_1_id]["in_transit_orders"][SKU2_ID])

        # There are a total of two trains in the configuration, and they have all been dispatched now.
        self.assertEqual(0, len(warehouse_1_distribution_unit._order_queues["train"]))
        order_3 = Order(warehouse_1, retailer_1, SKU2_ID, 3, "train", env.tick, None)
        warehouse_1_distribution_unit.place_order(order_3)
        consumer_unit._update_open_orders(retailer_1.id, SKU2_ID, 3)

        # After env.step runs, where tick is 4. order_1 will arrive at tick=5.
        # order_2 will arrive at tick=6.order_3 is expected to arrive at tick=8 under normal circumstances.
        env.step(None)
        self.assertEqual(
            [0, 1, 2, 0], list(env.metrics["products"][retailer_1.products[SKU2_ID].id]["pending_order_daily"]),
        )

        self.assertEqual(6, env.metrics["facilities"][retailer_1_id]["in_transit_orders"][SKU2_ID])

        self.assertEqual(0, env.metrics["facilities"][retailer_1_id]["in_transit_orders"][SKU3_ID])

        # After env.step runs, where tick is 5. order_1 arrives after env.step.
        # order_2 will arrive at tick=6.order_3 is expected to arrive at tick=8 under normal circumstances.
        env.step(None)
        self.assertEqual(
            [1, 2, 0, 0], list(env.metrics["products"][retailer_1.products[SKU2_ID].id]["pending_order_daily"]),
        )

        # After env.step runs, where tick is 6. order_2 arrives after env.step.
        # There are empty cars at this time, order_3 will arrive at tick = 11.
        env.step(None)
        self.assertEqual(
            [2, 0, 0, 0], list(env.metrics["products"][retailer_1.products[SKU2_ID].id]["pending_order_daily"]),
        )

        # When order_1 arrives at the next step, the in_transit_orders of retailer_1 should be the negative number
        # 1+2+3-1 of the arrival order of retailer_1.
        self.assertEqual(5, env.metrics["facilities"][retailer_1_id]["in_transit_orders"][SKU2_ID])

        # After env.step runs, where tick is 7. order_2 arrives after env.step.
        # There are empty cars at this time, order_3 will arrive at tick = 11.
        env.step(None)

        # When order_2 arrives at the next step, the in_transit_orders of retailer_1 should be the negative number
        # 1+2+3-1-2 of the arrival order of retailer_1.
        self.assertEqual(3, env.metrics["facilities"][retailer_1_id]["in_transit_orders"][SKU2_ID])

        order_4 = Order(warehouse_1, retailer_1, SKU2_ID, 4, "train", env.tick, None)
        warehouse_1_distribution_unit.place_order(order_4)
        consumer_unit._update_open_orders(retailer_1.id, SKU2_ID, 4)

        # After env.step runs, where tick is 8. order_3 will arrive at tick = 11.
        # order_4 is expected to arrive at tick=12 under normal circumstances.
        env.step(None)
        self.assertEqual(
            [0, 0, 0, 3], list(env.metrics["products"][retailer_1.products[SKU2_ID].id]["pending_order_daily"]),
        )

        # After env.step runs, where tick is 9. order_3 will arrive at tick = 11.
        # order_4 is expected to arrive at tick=12 under normal circumstances.
        env.step(None)
        self.assertEqual(
            [0, 0, 3, 4], list(env.metrics["products"][retailer_1.products[SKU2_ID].id]["pending_order_daily"]),
        )

        # After env.step runs, where tick is 10. order_3 will arrive at tick = 11.
        # order_4 is expected to arrive at tick=12 under normal circumstances.
        env.step(None)
        self.assertEqual(
            [0, 3, 4, 0], list(env.metrics["products"][retailer_1.products[SKU2_ID].id]["pending_order_daily"]),
        )

        # After env.step runs, where tick is 11. order_3 arrives after env.step.
        # order_4 is expected to arrive at tick=12 under normal circumstances.
        env.step(None)
        self.assertEqual(
            [3, 4, 0, 0], list(env.metrics["products"][retailer_1.products[SKU2_ID].id]["pending_order_daily"]),
        )

    def test_seller_state_only(self) -> None:
        """Test "sale_mean" and "_sale_hist """

        env = build_env("case_05", 600)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)
        store_001: FacilityBase = be.world._get_facility_by_name("Store_001")
        store_product_unit_sku1, store_product_unit_sku2, store_product_unit_sku3 = 1, 3, 2

        product_unit_sku1 = store_001.children[store_product_unit_sku1]
        product_unit_sku2 = store_001.children[store_product_unit_sku2]
        product_unit_sku3 = store_001.children[store_product_unit_sku3]
        assert isinstance(product_unit_sku1, ProductUnit)
        assert isinstance(product_unit_sku2, ProductUnit)
        assert isinstance(product_unit_sku3, ProductUnit)

        self.assertEqual([1, 1, 1, 1, 1, 1], product_unit_sku1.seller._sale_hist)
        self.assertEqual([2, 2, 2, 2, 2, 2], product_unit_sku2.seller._sale_hist)
        self.assertEqual([3, 3, 3, 3, 3, 3], product_unit_sku3.seller._sale_hist)

        env.step(None)

        # The demand in the data file should be added after env.step.
        self.assertEqual([1, 1, 1, 1, 1, 10], product_unit_sku1.seller._sale_hist)
        self.assertEqual([2, 2, 2, 2, 2, 100], product_unit_sku2.seller._sale_hist)
        self.assertEqual([3, 3, 3, 3, 3, 100], product_unit_sku3.seller._sale_hist)

        self.assertEqual(5, env.metrics["products"][store_001.products[SKU1_ID].id]["sale_mean"])
        self.assertEqual(5, env.metrics["products"][store_001.products[SKU1_ID].id]["demand_mean"])
        self.assertEqual(43.0, env.metrics["products"][store_001.products[SKU1_ID].id]["selling_price"])

        self.assertEqual(5*2+3, env.metrics["products"][store_001.products[SKU3_ID].id]["sale_mean"])
        self.assertEqual(5*2+3, env.metrics["products"][store_001.products[SKU3_ID].id]["demand_mean"])
        self.assertEqual(28.0, env.metrics["products"][store_001.products[SKU3_ID].id]["selling_price"])

        self.assertEqual(0+2, env.metrics["products"][store_001.products[SKU2_ID].id]["sale_mean"])
        self.assertEqual(0+2, env.metrics["products"][store_001.products[SKU2_ID].id]["demand_mean"])
        self.assertEqual(17.0, env.metrics["products"][store_001.products[SKU2_ID].id]["selling_price"])

        env.step(None)

        self.assertEqual([1, 1, 1, 1, 10, 20], product_unit_sku1.seller._sale_hist)
        self.assertEqual([2, 2, 2, 2, 100, 200], product_unit_sku2.seller._sale_hist)
        self.assertEqual([3, 3, 3, 3, 100, 200], product_unit_sku3.seller._sale_hist)

    def test_distribution_state_only(self) -> None:
        """Test the "pending_order" and "in_transit_orders" of the distribution unit."""
        env = build_env("case_05", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        supplier_3 = be.world._get_facility_by_name("Supplier_SKU3")
        distribution_unit = supplier_3.distribution
        warehouse_1 = be.world._get_facility_by_name("Warehouse_001")
        retailer_1: FacilityBase = be.world._get_facility_by_name("Retailer_001")
        consumer_unit = warehouse_1.products[3].consumer
        env.step(None)

        # There are 2 "train" in total, and 1 left after scheduling this order.
        order = Order(supplier_3, warehouse_1, SKU3_ID, 20, "train", env.tick, None)
        distribution_unit.place_order(order)
        consumer_unit._update_open_orders(warehouse_1, SKU3_ID, 20)
        self.assertEqual(1, len(distribution_unit._order_queues["train"]))
        self.assertEqual(20, sum([order.required_quantity for order in distribution_unit._order_queues["train"]]))
        supplier_3_id, warehouse_1_id, retailer_1_id = 1, 6, 13

        env.step(None)

        # vlt is greater than len(pending_order_len), which will cause the pending order to increase
        self.assertEqual(
            [0, 0, 20, 0], list(env.metrics["products"][warehouse_1.products[SKU3_ID].id]["pending_order_daily"]),
        )

        # add another order, it would be successfully scheduled.
        order = Order(supplier_3, warehouse_1, SKU3_ID, 25, "train", env.tick, None)
        distribution_unit.place_order(order)
        consumer_unit._update_open_orders(warehouse_1, SKU3_ID, 25)
        self.assertEqual(1, len(distribution_unit._order_queues["train"]))
        self.assertEqual(25, sum([order.required_quantity for order in distribution_unit._order_queues["train"]]))

        # 3rd order, will cause the pending order increase
        order_1 = Order(supplier_3, warehouse_1, SKU3_ID, 30, "train", env.tick, None)
        distribution_unit.place_order(order_1)
        consumer_unit._update_open_orders(warehouse_1, SKU3_ID, 30)

        self.assertEqual(2, len(distribution_unit._order_queues["train"]))
        self.assertEqual(55, sum([order.required_quantity for order in distribution_unit._order_queues["train"]]))
        self.assertEqual(55, env.metrics["facilities"][supplier_3_id]["pending_order"][SKU3_ID])
        self.assertEqual(55, distribution_unit._pending_product_quantity[SKU3_ID])

        warehouse_1_distribution_unit = warehouse_1.distribution

        order_2 = Order(warehouse_1, retailer_1, SKU3_ID, 5, "train", env.tick, None)
        warehouse_1_distribution_unit.place_order(order_2)
        consumer_unit._update_open_orders(warehouse_1, SKU3_ID, 5)

        order_3 = Order(warehouse_1, retailer_1, SKU3_ID, 5, "train", env.tick, None)
        warehouse_1_distribution_unit.place_order(order_3)
        consumer_unit._update_open_orders(warehouse_1, SKU3_ID, 5)

        self.assertEqual(5+5, env.metrics["facilities"][warehouse_1_id]["pending_order"][SKU3_ID])
        self.assertEqual(5+5, warehouse_1_distribution_unit._pending_product_quantity[SKU3_ID])

        consumer_unit._update_open_orders(warehouse_1, SKU3_ID, 5)
        warehouse_1_distribution_unit.place_order(order_2)
        self.assertEqual(15, env.metrics["facilities"][warehouse_1_id]["pending_order"][SKU3_ID])
        self.assertEqual(15, warehouse_1_distribution_unit._pending_product_quantity[SKU3_ID])

        # There is no place_order for the distribution of supplier_3, there should be no change
        self.assertEqual(55, env.metrics["facilities"][supplier_3_id]["pending_order"][SKU3_ID])
        self.assertEqual(55, distribution_unit._pending_product_quantity[SKU3_ID])

        start_tick = env.tick
        expected_supplier_tick = start_tick + 3

        while env.tick < expected_supplier_tick - 1:
            env.step(None)

        self.assertEqual(
            [20, 25, 0, 0], list(env.metrics["products"][warehouse_1.products[SKU3_ID].id]["pending_order_daily"]),
        )

        env.step(None)
        self.assertEqual(
            [25, 0, 0, 30], list(env.metrics["products"][warehouse_1.products[SKU3_ID].id]["pending_order_daily"]),
        )

        # will arrive at the end of this tick, still on the way.
        assert env.tick == expected_supplier_tick
        self.assertEqual(0, len(distribution_unit._order_queues["train"]))
        self.assertEqual(0, sum([order.required_quantity for order in distribution_unit._order_queues["train"]]))
        self.assertEqual(5, env.metrics["facilities"][warehouse_1_id]["pending_order"][SKU3_ID])
        self.assertEqual(5, warehouse_1_distribution_unit._pending_product_quantity[SKU3_ID])

        env.step(None)

        self.assertEqual(0, len(distribution_unit._order_queues["train"]))
        self.assertEqual(0, sum([order.required_quantity for order in distribution_unit._order_queues["train"]]))
        self.assertEqual(45, env.metrics["facilities"][warehouse_1_id]["in_transit_orders"][SKU3_ID])

        self.assertEqual(5, env.metrics["facilities"][warehouse_1_id]["pending_order"][SKU3_ID])
        self.assertEqual(5, warehouse_1_distribution_unit._pending_product_quantity[SKU3_ID])


if __name__ == "__main__":
    unittest.main()
