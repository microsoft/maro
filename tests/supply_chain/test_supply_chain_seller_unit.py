import os
import unittest
import numpy as np

from maro.simulator import Env
from maro.simulator.scenarios.supply_chain import FacilityBase
from maro.simulator.scenarios.supply_chain.business_engine import SupplyChainBusinessEngine


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
        Seller unit test:
            . initial state
            . with a customized seller unit
            . with built-in one
            . with dynamics sampler
        """

    def test_seller_unit_initial_states(self) -> None:
        """Test the initial states of sku3's SellerUnit of Retailer_001."""
        env = build_env("case_02", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        retailer_1 = be.world._get_facility_by_name("Retailer_001")
        seller_unit = retailer_1.products[SKU3_ID].seller

        self.assertEqual(SKU3_ID, seller_unit.product_id)

        # from configuration
        self.assertEqual(10, seller_unit._gamma)

        # initial value
        self.assertEqual(0, seller_unit._sold)
        self.assertEqual(0, seller_unit._demand)
        self.assertEqual(0, seller_unit._total_sold)

        # initial data model value
        self.assertEqual(seller_unit._sold, seller_unit.data_model.sold)
        self.assertEqual(seller_unit._demand, seller_unit.data_model.demand)
        self.assertEqual(seller_unit._total_sold, seller_unit.data_model.total_sold)

        env.reset()

        self.assertEqual(SKU3_ID, seller_unit.product_id)

        # from configuration
        self.assertEqual(10, seller_unit._gamma)

        self.assertEqual(0, seller_unit._sold)
        self.assertEqual(0, seller_unit._demand)
        self.assertEqual(0, seller_unit._total_sold)

        self.assertEqual(seller_unit._sold, seller_unit.data_model.sold)
        self.assertEqual(seller_unit._demand, seller_unit.data_model.demand)
        self.assertEqual(seller_unit._total_sold, seller_unit.data_model.total_sold)

    def test_seller_unit_demand_states(self) -> None:
        """Test the demand states of sku3's SellerUnit of Retailer_001."""
        env = build_env("case_02", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        retailer_1 = be.world._get_facility_by_name("Retailer_001")
        seller_unit = retailer_1.products[SKU3_ID].seller

        seller_node_index = seller_unit.data_model_index

        seller_nodes = env.snapshot_list["seller"]

        features = ("sold", "demand", "total_sold")
        IDX_SOLD, IDX_DEMAND, IDX_TOTAL_SOLD = 0, 1, 2

        self.assertEqual(10, retailer_1.skus[SKU3_ID].init_stock)
        self.assertEqual(10, retailer_1.storage._product_level[SKU3_ID])
        init_stock = 10

        env.step(None)

        # seller unit will try to count down the product quantity based on demand
        # default seller use gamma distribution on each tick

        # demand should be same with original
        self.assertEqual(seller_unit._demand, seller_unit.data_model.demand)

        actual_sold = min(seller_unit._demand, init_stock)
        # sold may be not same as demand, depend on remaining quantity in storage
        self.assertEqual(actual_sold, seller_unit._sold)
        self.assertEqual(actual_sold, seller_unit.data_model.sold)
        self.assertEqual(actual_sold, seller_unit._total_sold)
        self.assertEqual(actual_sold, seller_unit.data_model.total_sold)

        states = seller_nodes[env.frame_index:seller_node_index:features].flatten().astype(np.int)

        self.assertEqual(actual_sold, states[IDX_SOLD])
        self.assertEqual(seller_unit._demand, states[IDX_DEMAND])
        self.assertEqual(actual_sold, states[IDX_TOTAL_SOLD])

        # move to next step to check if state is correct
        env.step(None)

        # demand should be same with original
        self.assertEqual(seller_unit._demand, seller_unit.data_model.demand)

        actual_sold_2 = min(seller_unit._demand, init_stock - actual_sold)

        # sold may be not same as demand, depend on remaining quantity in storage
        self.assertEqual(actual_sold_2, seller_unit._sold)
        self.assertEqual(actual_sold_2, seller_unit.data_model.sold)
        self.assertEqual(actual_sold + actual_sold_2, seller_unit._total_sold)
        self.assertEqual(actual_sold + actual_sold_2, seller_unit.data_model.total_sold)

        states = seller_nodes[env.frame_index:seller_node_index:features].flatten().astype(np.int)

        self.assertEqual(actual_sold_2, states[IDX_SOLD])
        self.assertEqual(seller_unit._demand, states[IDX_DEMAND])
        self.assertEqual(actual_sold + actual_sold_2, states[IDX_TOTAL_SOLD])

    def test_seller_unit_customized(self) -> None:
        """Test customized SellerUnit of sku3 in Retailer_001."""
        env = build_env("case_03", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        retailer_1 = be.world._get_facility_by_name("Retailer_001")
        seller_unit = retailer_1.products[SKU3_ID].seller

        seller_node_index = seller_unit.data_model_index

        seller_nodes = env.snapshot_list["seller"]

        features = ("sold", "demand", "total_sold")
        IDX_SOLD, IDX_DEMAND, IDX_TOTAL_SOLD = 0, 1, 2

        self.assertEqual(SKU3_ID, seller_unit.product_id)

        # NOTE: this simple seller unit return demands that same as current tick

        # ######################### Tick 0 ##############################
        env.step(None)

        # Tick 0 will have demand == 0
        # from configuration
        self.assertEqual(0, seller_unit._sold)
        self.assertEqual(0, seller_unit._demand)
        self.assertEqual(0, seller_unit._total_sold)

        self.assertEqual(0, seller_unit.data_model.sold)
        self.assertEqual(0, seller_unit.data_model.demand)
        self.assertEqual(0, seller_unit.data_model.total_sold)

        is_done = False
        while not is_done:
            _, _, is_done = env.step(None)

        states = seller_nodes[:seller_node_index:features[IDX_DEMAND]].flatten().astype(np.int)

        # Check demand history, it should be same as tick
        self.assertListEqual([i for i in range(100)], list(states))

        # Check sold states. Since the init stock 10 = 1 + 2 + 3 + 4, sold value should be 0 after tick 4.
        states = seller_nodes[:seller_node_index:features[IDX_SOLD]].flatten().astype(np.int)
        self.assertListEqual([0, 1, 2, 3, 4] + [0] * 95, list(states))

        # Check total sold, should be: 0, 0 + 1, 0 + 1 + 2, 0 + 1 + 2 + 3, 0 + 1 + 2 + 3 + 4, 10, 10, ...
        states = seller_nodes[:seller_node_index:features[IDX_TOTAL_SOLD]].flatten().astype(np.int)
        self.assertListEqual([0, 1, 3, 6, 10] + [10] * 95, list(states))

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

        features = ("sold", "demand", "total_sold")
        IDX_SOLD, IDX_DEMAND, IDX_TOTAL_SOLD = 0, 1, 2

        self.assertEqual(20, seller_unit.product_id)

        # NOTE: this simple seller unit return demands that same as current tick

        # Tick 0 will have demand == 25.first row of data after preprocessing data.
        # from sample_preprocessed.csv
        self.assertEqual(10, seller_unit._sold)
        self.assertEqual(10, seller_unit._demand)
        self.assertEqual(10, seller_unit._total_sold)

        self.assertEqual(10, seller_unit.data_model.sold)
        self.assertEqual(10, seller_unit.data_model.demand)
        self.assertEqual(10, seller_unit.data_model.total_sold)

        expected_tick = 5
        while env.tick < expected_tick - 1:
            env.step(None)

        states = seller_nodes[:seller_node_index:features[IDX_SOLD]].flatten().astype(np.int)
        self.assertListEqual([10, 20, 30, 40, 50], list(states))


if __name__ == '__main__':
    unittest.main()
