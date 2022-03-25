import os
import unittest
from typing import Optional

import numpy as np

from maro.simulator import Env
from maro.simulator.scenarios.supply_chain import (
    ConsumerAction, ConsumerUnit, DistributionUnit, FacilityBase, ManufactureAction, SellerUnit, StorageUnit,
    VehicleUnit,
)
from maro.simulator.scenarios.supply_chain.business_engine import SupplyChainBusinessEngine
from maro.simulator.scenarios.supply_chain.units.order import Order
from maro.simulator.scenarios.supply_chain.units.storage import AddStrategy


def build_env(case_name: str, durations: int):
    case_folder = os.path.join("tests", "data", "supply_chain", case_name)

    # config_path = os.path.join(case_folder, "config.yml")

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


class MyTestCase(unittest.TestCase):
    """
    manufacture unit testing:

    1. with input sku
        . meet the storage limitation
        . not meet the storage limitation
        . with enough source sku
        . without enough source sku
        . with product rate
        . without product rate
    2. without input sku
        . meet the storage limitation
        . not meet the storage limitation
        . with product rate
        . without product rate

    """

    def test_manufacture_meet_storage_limitation(self):
        """Test sku3 manufacturing."""
        env = build_env("case_01", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        storage_nodes = env.snapshot_list["storage"]
        storage_features = ("id", "facility_id", "capacity", "remaining_space")

        manufacture_nodes = env.snapshot_list["manufacture"]
        manufacture_number = len(manufacture_nodes)
        manufacture_features = (
            "id", "facility_id", "manufacture_quantity", "product_id", "product_unit_cost"
        )

        # ############################### TICK: 0 ######################################

        # tick 0 passed, no product manufacturing.
        env.step(None)

        states = manufacture_nodes[
            env.frame_index::manufacture_features
        ].flatten().reshape(manufacture_number, -1).astype(np.int)

        # try to find which one is sku3 manufacture unit.
        sku3_data_model_index: Optional[int] = None
        sku3_manufacture_id: Optional[int] = None
        sku3_facility_id: Optional[int] = None
        for index, state in enumerate(states):
            # Id of sku3 is 3.
            if state[3] == SKU3_ID:
                sku3_data_model_index = index
                sku3_manufacture_id = state[0]
                sku3_facility_id = state[1]
        self.assertTrue(all([
            sku3_data_model_index is not None,
            sku3_manufacture_id is not None,
            sku3_facility_id is not None
        ]))

        # try to find sku3's storage from env.summary
        sku3_facility_info = env.summary["node_mapping"]["facilities"][sku3_facility_id]
        sku3_storage_index = sku3_facility_info["units"]["storage"]["node_index"]

        storage_states = storage_nodes[env.frame_index:sku3_storage_index:storage_features].flatten().astype(np.int)

        # there should be 80 units been taken at the beginning according to the config file.
        # so remaining space should be 20
        self.assertEqual(20, storage_states[3])
        # capacity is 100 by config
        self.assertEqual(100, storage_states[2])

        product_dict = get_product_dict_from_storage(env, env.frame_index, sku3_storage_index)

        # The product quantity should be same as configuration at beginning.
        # 80 sku3
        self.assertEqual(80, product_dict[SKU3_ID])

        # all the id is greater than 0
        self.assertGreater(sku3_manufacture_id, 0)

        # ############################### TICK: 1 ######################################

        # pass an action to start manufacturing for this tick.
        action = ManufactureAction(sku3_manufacture_id, 1)

        env.step([action])

        states = manufacture_nodes[env.frame_index:sku3_data_model_index:manufacture_features].flatten().astype(np.int)

        # Sku3 produce rate is 1 per tick, so manufacture_quantity should be 1.
        self.assertEqual(1, states[2])

        storage_states = storage_nodes[env.frame_index:sku3_storage_index:storage_features].flatten().astype(np.int)

        # now remaining space should be 19
        self.assertEqual(19, storage_states[3])

        product_dict = get_product_dict_from_storage(env, env.frame_index, sku3_storage_index)

        # sku3 quantity should be 80 + 1
        self.assertEqual(80 + 1, product_dict[SKU3_ID])

        # ############################### TICK: 2 ######################################

        # leave the action as none will cause manufacture unit stop manufacturing.
        env.step(None)

        states = manufacture_nodes[env.frame_index:sku3_data_model_index:manufacture_features].flatten().astype(np.int)

        # so manufacture_quantity should be 0
        self.assertEqual(0, states[2])

        product_dict = get_product_dict_from_storage(env, env.frame_index, sku3_storage_index)

        # sku3 quantity should be same as last tick
        self.assertEqual(80 + 1, product_dict[SKU3_ID])

        # let is generate 20, but actually it can only procedure 19 because the storage will reach the limitation
        env.step([ManufactureAction(sku3_manufacture_id, 20)])

        states = manufacture_nodes[env.frame_index:sku3_data_model_index:manufacture_features].flatten().astype(np.int)

        # so manufacture_number should be 19 instead 20
        self.assertEqual(19, states[2])

        storage_states = storage_nodes[env.frame_index:sku3_storage_index:storage_features].flatten().astype(np.int)

        # now remaining space should be 0
        self.assertEqual(0, storage_states[3])

        product_dict = get_product_dict_from_storage(env, env.frame_index, sku3_storage_index)

        # sku3 quantity should be 100
        self.assertEqual(80 + 1 + 19, product_dict[SKU3_ID])

    def test_manufacture_meet_source_lack(self):
        """Test sku4 manufacturing, this sku supplier does not have enough source material at the begging
            , so it cannot produce anything without consumer purchase."""
        env = build_env("case_01", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        storage_nodes = env.snapshot_list["storage"]
        storage_features = ("id", "facility_id", "capacity", "remaining_space")

        manufacture_nodes = env.snapshot_list["manufacture"]
        manufacture_number = len(manufacture_nodes)
        manufacture_features = (
            "id", "facility_id", "manufacture_quantity", "product_id", "product_unit_cost"
        )

        # ############################### TICK: 0 ######################################

        # tick 0 passed, no product manufacturing.
        env.step(None)

        states = manufacture_nodes[
            env.frame_index::manufacture_features
        ].flatten().reshape(manufacture_number, -1).astype(np.int)

        # try to find which one is sku3 manufacture unit.
        sku4_data_model_index: Optional[int] = None
        sku4_manufacture_id: Optional[int] = None
        sku4_facility_id: Optional[int] = None
        for index, state in enumerate(states):
            # Id of sku4 is 4.
            if state[3] == SKU4_ID:
                sku4_data_model_index = index
                sku4_manufacture_id = state[0]
                sku4_facility_id = state[1]
        self.assertTrue(all([
            sku4_data_model_index is not None,
            sku4_manufacture_id is not None,
            sku4_facility_id is not None
        ]))

        # try to find sku4's storage from env.summary
        sku4_facility_info = env.summary["node_mapping"]["facilities"][sku4_facility_id]
        sku4_storage_index = sku4_facility_info["units"]["storage"]["node_index"]

        # the storage should be same as initialized (50 + 0).
        storage_states = storage_nodes[env.frame_index:sku4_storage_index:storage_features].flatten().astype(np.int)

        # capacity is same as configured.
        self.assertEqual(200, storage_states[2])

        # remaining space should be capacity - (50+0)
        self.assertEqual(200 - (50 + 0), storage_states[3])

        # no manufacture number as we have not pass any action
        manufature_states = manufacture_nodes[
            env.frame_index:sku4_data_model_index:manufacture_features
        ].flatten().astype(np.int)

        # manufacture_quantity should be 0
        self.assertEqual(0, manufature_states[2])

        # output product id should be same as configured.
        self.assertEqual(4, manufature_states[3])

        # product unit cost should be same as configured.
        self.assertEqual(4, manufature_states[4])

        product_dict = get_product_dict_from_storage(env, env.frame_index, sku4_storage_index)

        # 50 sku4 at beginning
        self.assertEqual(50, product_dict[SKU4_ID])

        # 0 sku2
        self.assertEqual(0, product_dict[SKU2_ID])

        # ############################### TICK: 1 - end ######################################

        is_done = False

        while not is_done:
            # push to the end, the storage should not changed, no matter what production rate we give it.
            _, _, is_done = env.step([ManufactureAction(sku4_manufacture_id, 10)])

        manufature_states = manufacture_nodes[
            env.frame_index:sku4_data_model_index:manufacture_features
        ].flatten().astype(np.int)

        # manufacture_quantity should be 0
        self.assertEqual(0, manufature_states[2])

        # output product id should be same as configured.
        self.assertEqual(SKU4_ID, manufature_states[3])

        # product unit cost should be same as configured.
        self.assertEqual(4, manufature_states[4])

        product_dict = get_product_dict_from_storage(env, env.frame_index, sku4_storage_index)

        # 50 sku4 at beginning
        self.assertEqual(50, product_dict[SKU4_ID])

        # 0 sku2
        self.assertEqual(0, product_dict[SKU2_ID])

    def test_manufacture_meet_avg_storage_limitation(self):
        """Test on sku1, it is configured with nearly full initial states."""

        env = build_env("case_01", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        storage_nodes = env.snapshot_list["storage"]
        storage_features = ("id", "facility_id", "capacity", "remaining_space")

        manufacture_nodes = env.snapshot_list["manufacture"]
        manufacture_number = len(manufacture_nodes)
        manufacture_features = (
            "id", "facility_id", "manufacture_quantity", "product_id", "product_unit_cost"
        )

        # ############################### TICK: 0 ######################################

        # tick 0 passed, no product manufacturing, verified in above case, pass checking it here.
        env.step(None)

        states = manufacture_nodes[
            env.frame_index::manufacture_features
        ].flatten().reshape(manufacture_number, -1).astype(np.int)
        # try to find which one is sku3 manufacture unit.
        sku1_data_model_index: Optional[int] = None
        sku1_manufacture_id: Optional[int] = None
        sku1_facility_id: Optional[int] = None
        for index, state in enumerate(states):
            # Id of sku1 is 1.
            if state[3] == SKU1_ID:
                sku1_data_model_index = index
                sku1_manufacture_id = state[0]
                sku1_facility_id = state[1]
        self.assertTrue(all([
            sku1_data_model_index is not None,
            sku1_manufacture_id is not None,
            sku1_facility_id is not None
        ]))

        sku1_facility_info = env.summary["node_mapping"]["facilities"][sku1_facility_id]
        sku1_storage_index = sku1_facility_info["units"]["storage"]["node_index"]

        # ############################### TICK: 1 ######################################

        # ask sku1 manufacture start manufacturing, rate is 10.
        env.step([ManufactureAction(sku1_manufacture_id, 10)])

        storage_states = storage_nodes[env.frame_index:sku1_storage_index:storage_features].flatten().astype(np.int)
        manufacture_states = manufacture_nodes[
                             env.frame_index:sku1_data_model_index:manufacture_features].flatten().astype(np.int)

        # we can produce 4 sku1, as it will meet storage avg limitation per sku
        self.assertEqual(4, manufacture_states[2])

        # so storage remaining space should be 200 - ((96 + 4) + (100 - 4*2))
        self.assertEqual(200 - ((96 + 4) + (100 - 4 * 2)), storage_states[3])

        product_dict = get_product_dict_from_storage(env, env.frame_index, sku1_storage_index)

        # The product quantity of sku1 should 100, just reach the avg storage capacity limitation
        self.assertEqual(100, product_dict[SKU1_ID])

        # 4 sku1 cost 4*2 source material (sku3)
        self.assertEqual(100 - 4 * 2, product_dict[SKU3_ID])

        # ############################### TICK: 1 ######################################

        # then fix the product rate to 20 every tick, but the manufacture will do nothing, as we have to enough space

        is_done = False

        while not is_done:
            _, _, is_done = env.step([ManufactureAction(sku1_storage_index, 20)])

        storage_states = storage_nodes[env.frame_index:sku1_storage_index:storage_features].flatten().astype(np.int)
        manufacture_states = manufacture_nodes[
                             env.frame_index:sku1_data_model_index:manufacture_features].flatten().astype(np.int)

        # but manufacture number is 0
        self.assertEqual(0, manufacture_states[2])

        # so storage remaining space should be 200 - ((96 + 4) + (100 - 4*2))
        self.assertEqual(200 - ((96 + 4) + (100 - 4 * 2)), storage_states[3])

        product_dict = get_product_dict_from_storage(env, env.frame_index, sku1_storage_index)

        # The product quantity of sku1 should 100, just reach the avg storage capacity limitation
        self.assertEqual(100, product_dict[SKU1_ID])

        # 4 sku1 cost 4*2 source material (sku3)
        self.assertEqual(100 - 4 * 2, product_dict[SKU3_ID])

    """
    Storage test:

    . take available
        . enough
        . not enough
    . try add products
        . meet whole storage capacity limitation
            . fail if all
            . not fail if all
        . enough space
    . try take products
        . have enough
        . not enough
    . get product quantity

    """

    def test_storage_take_available(self):
        env = build_env("case_01", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        env.step(None)

        storage_nodes = env.snapshot_list["storage"]

        # find first storage unit id
        storage_unit_id = storage_nodes[env.frame_index:0:"id"].flatten().astype(np.int)[0]

        # get the unit reference from env internal
        storage_unit: StorageUnit = be.world.get_entity_by_id(storage_unit_id)

        init_product_dict = get_product_dict_from_storage(env, env.frame_index, 0)

        # call take_available for each product in storage.
        products_taken = {}
        for product_id, product_quantity in init_product_dict.items():
            quantity = np.random.randint(0, product_quantity)
            actual_quantity = storage_unit.take_available(product_id, quantity)

            # we should get the quantity we want.
            self.assertEqual(quantity, actual_quantity)

            products_taken[product_id] = quantity

        # check if internal state correct
        for product_id, quantity in products_taken.items():
            remaining_quantity = storage_unit._product_level[product_id]

            self.assertEqual(init_product_dict[product_id] - quantity, remaining_quantity)

        # call env.step will cause states write into snapshot
        env.step(None)

        product_dict = get_product_dict_from_storage(env, env.frame_index, 0)

        for product_id, quantity in products_taken.items():
            remaining_quantity = product_dict[product_id]

            self.assertEqual(init_product_dict[product_id] - quantity, remaining_quantity)

        # then take more than exist quantity for 1st product(sku)
        lot_taken_product_id, lot_taken_product_quantity = product_dict.popitem()

        lot_taken_product_quantity += 100

        actual_quantity = storage_unit.take_available(lot_taken_product_id, lot_taken_product_quantity)

        # we should get all available
        self.assertEqual(actual_quantity, lot_taken_product_quantity - 100)

        # take snapshot
        env.step(None)

        product_dict = get_product_dict_from_storage(env, env.frame_index, 0)

        # the product quantity should be 0, as we took all available
        self.assertEqual(0, product_dict[lot_taken_product_id])

    def test_storage_try_add_products(self):
        # TODO: Add test for other strategies.
        """
        NOTE:
            try_add_products method do not check avg storage capacity checking, so we will ignore it here.

        """
        env = build_env("case_01", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        env.step(None)

        storage_nodes = env.snapshot_list["storage"]
        storage_features = ("id", "capacity", "remaining_space")

        # find first storage unit id
        storage_unit_id = storage_nodes[env.frame_index:0:"id"].flatten().astype(np.int)[0]

        # get the unit reference from env internal
        storage_unit: StorageUnit = be.world.get_entity_by_id(storage_unit_id)

        storage_states = storage_nodes[env.frame_index:0:storage_features].flatten().astype(np.int)

        capacity = storage_states[1]
        init_remaining_space = storage_states[2]

        init_product_dict = get_product_dict_from_storage(env, env.frame_index, 0)

        # try put products out of capacity with all_or_nothing == True
        products_to_put = {}

        avg_max_product_quantity = init_remaining_space // len(init_product_dict)

        for product_id in init_product_dict.keys():
            products_to_put[product_id] = avg_max_product_quantity + 1

        result = storage_unit.try_add_products(products_to_put, add_strategy=AddStrategy.IgnoreUpperBoundAllOrNothing)

        # the method will return an empty dictionary if fail to add
        self.assertEqual(0, len(result))

        # so remaining space should not change
        self.assertEqual(init_remaining_space, storage_unit.remaining_space)

        # each product quantity should be same as before
        for product_id, product_quantity in init_product_dict.items():
            self.assertEqual(product_quantity, storage_unit._product_level[product_id])

        # if we set all_or_nothing=False, then part of the product will be added to storage, and cause remaining
        # space being 0
        storage_unit.try_add_products(products_to_put, add_strategy=AddStrategy.IgnoreUpperBoundAddInOrder)

        self.assertEqual(0, storage_unit.remaining_space)

        # take snapshot
        env.step(None)

        storage_states = storage_nodes[env.frame_index:0:storage_features].flatten().astype(np.int)

        # remaining space in snapshot should be 0
        self.assertEqual(0, storage_states[2])

        product_dict = get_product_dict_from_storage(env, env.frame_index, 0)

        # total product quantity should be same as capacity
        self.assertEqual(capacity, sum(product_dict.values()))

        # ###################################################
        # ###################################################
        # reset the env for next case
        env.reset()

        # check the state after reset
        self.assertEqual(capacity, storage_unit.capacity)
        self.assertEqual(init_remaining_space, storage_unit.remaining_space)

        for product_id, product_quantity in init_product_dict.items():
            self.assertEqual(product_quantity, storage_unit._product_level[product_id])

    def test_storage_try_take_products(self):
        env = build_env("case_01", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        env.step(None)

        storage_nodes = env.snapshot_list["storage"]
        storage_features = ("id", "capacity", "remaining_space")

        # find first storage unit id
        storage_unit_id = storage_nodes[env.frame_index:0:"id"].flatten().astype(np.int)[0]

        # get the unit reference from env internal
        storage_unit: StorageUnit = be.world.get_entity_by_id(storage_unit_id)

        storage_states = storage_nodes[env.frame_index:0:storage_features].flatten().astype(np.int)

        capacity = storage_states[1]
        init_remaining_space = storage_states[2]

        init_product_dict = get_product_dict_from_storage(env, env.frame_index, 0)

        product_to_take = {}

        for product_id, product_quantity in init_product_dict.items():
            product_to_take[product_id] = product_quantity + 1

        # which this setting, it will return false, as no enough product for ous
        self.assertFalse(storage_unit.try_take_products(product_to_take))

        # so remaining space and product quantity should same as before
        self.assertEqual(init_remaining_space, storage_unit.remaining_space)

        for product_id, product_quantity in init_product_dict.items():
            self.assertEqual(product_quantity, storage_unit._product_level[product_id])

        # try to get all products
        for product_id, product_quantity in product_to_take.items():
            product_to_take[product_id] = product_quantity - 1

        self.assertTrue(storage_unit.try_take_products(product_to_take))

        # now the remaining space should be same as capacity as we take all
        self.assertEqual(capacity, storage_unit.remaining_space)

        # take snapshot
        env.step(None)

        storage_states = storage_nodes[env.frame_index:0:storage_features].flatten().astype(np.int)

        # remaining space should be same as capacity in snapshot
        self.assertEqual(storage_states[1], storage_states[2])

    def test_storage_get_product_quantity(self):
        env = build_env("case_01", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        env.step(None)

        storage_nodes = env.snapshot_list["storage"]

        # find first storage unit id
        storage_unit_id = storage_nodes[env.frame_index:0:"id"].flatten().astype(np.int)[0]

        # get the unit reference from env internal
        storage_unit: StorageUnit = be.world.get_entity_by_id(storage_unit_id)

        init_product_dict = get_product_dict_from_storage(env, env.frame_index, 0)

        # The product quantity in object should be same with states
        for product_id, product_quantity in init_product_dict.items():
            self.assertEqual(product_quantity, storage_unit._product_level[product_id])

        # should not change even after reset
        env.reset()
        env.step(None)

        init_product_dict = get_product_dict_from_storage(env, env.frame_index, 0)

        # The product quantity in object should be same with states
        for product_id, product_quantity in init_product_dict.items():
            self.assertEqual(product_quantity, storage_unit._product_level[product_id])

    """

    Consumer test:

    . initial state
    . state after reset
    . set_action directly from code
    . set_action by env.step
    . call on_order_reception directly to simulation order arrived
    . call update_open_orders directly

    """

    def test_consumer_init_state(self):
        """
        NOTE: we will use consumer on Supplier_SKU1, as it contains a source for sku3 (Supplier_SKU3)
        """
        env = build_env("case_01", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        # print(env.summary)
        # we can get the consumer from env.summary

        # NOTE: though we are test with sku1, but the consumer is for sku3, as it is the source material from source
        sku3_consumer_unit: Optional[ConsumerUnit] = None
        sku3_product_unit_id: Optional[int] = None
        sku3_consumer_unit_id: Optional[int] = None

        for _, facility_detail in env.summary["node_mapping"]["facilities"].items():
            if facility_detail["name"] == "Supplier_SKU1":
                # try to find sku3 consumer
                sku3_consumer_unit_id = facility_detail["units"]["products"][SKU3_ID]["consumer"]["id"]

                sku3_consumer_unit = be.world.get_entity_by_id(sku3_consumer_unit_id)
                sku3_product_unit_id = facility_detail["units"]["products"][SKU3_ID]["id"]

        sku3_consumer_data_model_index = env.summary["node_mapping"]["unit_mapping"][sku3_consumer_unit_id][1]

        # check initial state
        self.assertEqual(0, sku3_consumer_unit._received)
        self.assertEqual(0, sku3_consumer_unit._purchased)
        self.assertEqual(0, sku3_consumer_unit._order_product_cost)
        self.assertEqual(SKU3_ID, sku3_consumer_unit.product_id)

        # check data model state
        # order cost from configuration
        self.assertEqual(200, sku3_consumer_unit.data_model.order_cost)
        self.assertEqual(0, sku3_consumer_unit.data_model.total_purchased)
        self.assertEqual(0, sku3_consumer_unit.data_model.total_received)

        # NOTE: 0 is an invalid(initial) id
        self.assertEqual(SKU3_ID, sku3_consumer_unit.data_model.product_id)
        self.assertEqual(sku3_consumer_unit_id, sku3_consumer_unit.data_model.id)
        self.assertEqual(sku3_product_unit_id, sku3_consumer_unit.data_model.product_unit_id)
        self.assertEqual(0, sku3_consumer_unit.data_model.order_quantity)
        self.assertEqual(0, sku3_consumer_unit.data_model.purchased)
        self.assertEqual(0, sku3_consumer_unit.data_model.received)
        self.assertEqual(0, sku3_consumer_unit.data_model.order_product_cost)

        # check sources
        for source_facility_id in sku3_consumer_unit.sources:
            source_facility: FacilityBase = be.world.get_facility_by_id(source_facility_id)

            # check if source facility contains the sku3 config
            self.assertTrue(SKU3_ID in source_facility.skus)

        env.step(None)

        # check state
        features = (
            "id",
            "facility_id",
            "product_id",
            "order_cost",
            "total_purchased",
            "total_received",
            "order_quantity",
            "purchased",
            "received",
            "order_product_cost"
        )

        consumer_nodes = env.snapshot_list["consumer"]

        states = consumer_nodes[env.frame_index:sku3_consumer_data_model_index:features].flatten().astype(np.int)

        # Nothing happened at tick 0, so most states will be 0
        self.assertTrue((states[4:] == 0).all())

        self.assertEqual(sku3_consumer_unit_id, states[0])
        self.assertEqual(SKU3_ID, states[2])

        env.reset()
        env.step(None)

        states = consumer_nodes[env.frame_index:sku3_consumer_data_model_index:features].flatten().astype(np.int)

        # Nothing happened at tick 0, so most states will be 0
        self.assertTrue((states[4:] == 0).all())

        self.assertEqual(sku3_consumer_unit_id, states[0])
        self.assertEqual(SKU3_ID, states[2])

    def test_consumer_action(self):
        env = build_env("case_01", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        sku3_consumer_unit_id: Optional[int] = None
        sku3_consumer_unit: Optional[ConsumerUnit] = None
        sku3_supplier_facility_id: Optional[int] = None

        for facility_id, facility_detail in env.summary["node_mapping"]["facilities"].items():
            if facility_detail["name"] == "Supplier_SKU1":
                sku3_consumer_unit_id = facility_detail["units"]["products"][SKU3_ID]["consumer"]["id"]
                sku3_consumer_unit = be.world.get_entity_by_id(sku3_consumer_unit_id)
            if facility_detail["name"] == "Supplier_SKU3":
                sku3_supplier_facility_id = facility_detail["id"]
        self.assertTrue(all([
            sku3_consumer_unit_id is not None,
            sku3_consumer_unit is not None,
            sku3_supplier_facility_id is not None
        ]))

        sku3_consumer_data_model_index = env.summary["node_mapping"]["unit_mapping"][sku3_consumer_unit_id][1]

        # zero quantity will be ignore
        action_with_zero = ConsumerAction(sku3_consumer_unit_id, SKU3_ID, sku3_supplier_facility_id, 0, 1)

        action = ConsumerAction(sku3_consumer_unit_id, SKU3_ID, sku3_supplier_facility_id, 10, 1)

        sku3_consumer_unit.set_action(action_with_zero)

        env.step(None)

        features = (
            "id",
            "facility_id",
            "product_id",
            "order_cost",
            "total_purchased",
            "total_received",
            "product_id",
            "order_quantity",
            "purchased",
            "received",
            "order_product_cost"
        )

        consumer_nodes = env.snapshot_list["consumer"]

        states = consumer_nodes[env.frame_index:sku3_consumer_data_model_index:features].flatten().astype(np.int)

        # Nothing happened at tick 0, at the action will be recorded
        self.assertEqual(action_with_zero.product_id, states[6])
        self.assertEqual(action_with_zero.quantity, states[8])

        self.assertEqual(sku3_consumer_unit_id, states[0])
        self.assertEqual(SKU3_ID, states[2])

        # NOTE: we cannot set_action directly here, as post_step will clear the action before starting next tick
        env.step([action])

        self.assertEqual(action.quantity, sku3_consumer_unit._purchased)
        self.assertEqual(0, sku3_consumer_unit._received)

        states = consumer_nodes[env.frame_index:sku3_consumer_data_model_index:features].flatten().astype(np.int)

        # action field should be recorded
        self.assertEqual(action.product_id, states[6])
        self.assertEqual(action.quantity, states[8])

        # total purchased should be same as purchased at this tick.
        self.assertEqual(action.quantity, states[4])

        # no received now
        self.assertEqual(0, states[5])

        # purchased same as quantity
        self.assertEqual(action.quantity, states[8])

        # no receives
        self.assertEqual(0, states[9])

        # same action for next step, so total_XXX will be changed to double
        env.step([action])

        states = consumer_nodes[env.frame_index:sku3_consumer_data_model_index:features].flatten().astype(np.int)

        # action field should be recorded
        self.assertEqual(action.product_id, states[6])
        self.assertEqual(action.quantity, states[8])

        # total purchased should be same as purchased at this tick.
        self.assertEqual(action.quantity * 2, states[4])

        # no received now
        self.assertEqual(0, states[5])

        # purchased same as quantity
        self.assertEqual(action.quantity, states[8])

        # no receives
        self.assertEqual(0, states[9])

    def test_consumer_on_order_reception(self):
        env = build_env("case_01", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        sku3_consumer_unit_id: Optional[int] = None
        sku3_consumer_unit: Optional[ConsumerUnit] = None
        sku3_supplier_facility_id: Optional[int] = None

        for facility_id, facility_detail in env.summary["node_mapping"]["facilities"].items():
            if facility_detail["name"] == "Supplier_SKU1":
                sku3_consumer_unit_id = facility_detail["units"]["products"][SKU3_ID]["consumer"]["id"]
                sku3_consumer_unit = be.world.get_entity_by_id(sku3_consumer_unit_id)
            if facility_detail["name"] == "Supplier_SKU3":
                sku3_supplier_facility_id = facility_detail["id"]
        self.assertTrue(all([
            sku3_consumer_unit_id is not None,
            sku3_consumer_unit is not None,
            sku3_supplier_facility_id is not None
        ]))

        action = ConsumerAction(sku3_consumer_unit_id, SKU3_ID, sku3_supplier_facility_id, 10, 1)

        # 1st step must none action
        env.step(None)

        env.step([action])

        # simulate purchased product is arrived by vehicle unit
        sku3_consumer_unit.on_order_reception(sku3_supplier_facility_id, SKU3_ID, 10, 10)

        # now all order is done
        self.assertEqual(0, sku3_consumer_unit._open_orders[sku3_supplier_facility_id][SKU3_ID])
        self.assertEqual(10, sku3_consumer_unit._received)

        env.step(None)

        # NOTE: we cannot test the received state by calling on_order_reception directly,
        # as it will be cleared by env.step, do it on vehicle unit test.

    """
    Vehicle unit test:

    . initial state
    . if vehicle arrive at destination within special vlt
    . schedule job
    . try_load until patient <= 0 to cancel the schedule
    . try_load until patient > 0 to load order
    . try_unload
        . target storage cannot take all
        . target storage can take all
    """

    def test_vehicle_unit_state(self):
        env = build_env("case_02", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        # try to find first vehicle unit we meet
        vehicle_unit: Optional[VehicleUnit] = None

        for unit_id, info in env.summary["node_mapping"]["unit_mapping"].items():
            if info[0] == "vehicle":
                vehicle_unit = be.world.get_entity_by_id(unit_id)

                break
        self.assertTrue(vehicle_unit is not None)

        # check initial state according to configuration file
        self.assertEqual(10, vehicle_unit._max_patient)

        self.assertEqual(0, vehicle_unit.requested_quantity)
        # not destination at first
        self.assertIsNone(vehicle_unit._destination)
        # no path
        self.assertIsNone(vehicle_unit._path)
        # no product
        self.assertEqual(0, vehicle_unit.product_id)
        # no steps
        self.assertEqual(0, vehicle_unit._remaining_steps)
        #
        self.assertEqual(0, vehicle_unit.payload)
        #
        self.assertEqual(0, vehicle_unit._location)
        #
        self.assertEqual(0, vehicle_unit._velocity)

        # state in frame
        self.assertEqual(0, vehicle_unit.data_model.payload)
        self.assertEqual(12, vehicle_unit.data_model.unit_transport_cost)

        # reset to check again
        env.step(None)
        env.reset()

        # check initial state according to configuration file
        self.assertEqual(10, vehicle_unit._max_patient)

        # not destination at first
        self.assertIsNone(vehicle_unit._destination)
        # no path
        self.assertIsNone(vehicle_unit._path)
        # no product
        self.assertEqual(0, vehicle_unit.product_id)
        # no steps
        self.assertEqual(0, vehicle_unit._remaining_steps)
        #
        self.assertEqual(0, vehicle_unit.payload)
        #
        self.assertEqual(0, vehicle_unit._location)
        #
        self.assertEqual(0, vehicle_unit._velocity)
        #
        self.assertEqual(0, vehicle_unit.requested_quantity)

        # state in frame
        self.assertEqual(0, vehicle_unit.data_model.payload)
        self.assertEqual(12, vehicle_unit.data_model.unit_transport_cost)

    def test_vehicle_unit_schedule(self):
        env = build_env("case_02", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        # try to find first vehicle unit of Supplier
        vehicle_unit: Optional[VehicleUnit] = None
        dest_facility: Optional[FacilityBase] = None

        for _, info in env.summary["node_mapping"]["facilities"].items():
            if info["name"] == "Supplier_SKU3":
                for v in info["units"]["distribution"]["children"]:
                    vehicle_unit = be.world.get_entity_by_id(v["id"])

            if info["name"] == "Warehouse_001":
                dest_facility = be.world.get_facility_by_id(info["id"])
        self.assertTrue(all([vehicle_unit is not None, dest_facility is not None]))

        # make sure the upstream in the only one supplier in config
        self.assertEqual(1, len(dest_facility.upstreams))
        self.assertEqual(1, len(dest_facility.upstreams[SKU3_ID]))

        # schedule job vehicle unit manually, from supplier to warehouse
        vehicle_unit.schedule(dest_facility, SKU3_ID, 20, 2)

        # step to take snapshot
        env.step(None)

        vehicle_nodes = env.snapshot_list["vehicle"]

        # check internal states
        self.assertEqual(dest_facility, vehicle_unit._destination)
        self.assertEqual(SKU3_ID, vehicle_unit.product_id)
        self.assertEqual(20, vehicle_unit.requested_quantity)
        self.assertEqual(1, vehicle_unit._velocity)
        self.assertEqual(2, vehicle_unit._remaining_steps)

        features = (
            "id",
            "facility_id",
            "payload",
            "unit_transport_cost"
        )

        states = vehicle_nodes[env.frame_index:vehicle_unit.data_model_index:features].flatten().astype(np.int)

        # source id
        self.assertEqual(vehicle_unit.facility.id, states[1])
        # payload should be 20, as we already env.step
        self.assertEqual(20, states[2])

        # push the vehicle on the way
        env.step(None)

        states = vehicle_nodes[env.frame_index:vehicle_unit.data_model_index:features].flatten().astype(np.int)

        # payload
        self.assertEqual(20, states[2])

        env.step(None)
        env.step(None)

        # next step vehicle will try to unload the products
        env.step(None)

        states = vehicle_nodes[env.frame_index:vehicle_unit.data_model_index:features].flatten().astype(np.int)

        # the product is unloaded, vehicle states will be reset to initial
        # not destination at first
        self.assertIsNone(vehicle_unit._destination)
        self.assertIsNone(vehicle_unit._path)
        self.assertEqual(0, vehicle_unit.product_id)
        self.assertEqual(0, vehicle_unit._remaining_steps)
        self.assertEqual(0, vehicle_unit.payload)
        self.assertEqual(0, vehicle_unit._location)
        self.assertEqual(0, vehicle_unit._velocity)
        self.assertEqual(0, vehicle_unit.requested_quantity)

        # check states
        self.assertEqual(0, states[2])
        self.assertEqual(12, vehicle_unit.data_model.unit_transport_cost)

    def test_vehicle_unit_no_patient(self):
        """
        NOTE: with patient is tried in above case after schedule the job
        """
        env = build_env("case_02", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        # try to find first vehicle unit of Supplier
        vehicle_unit: Optional[VehicleUnit] = None
        dest_facility: Optional[FacilityBase] = None

        for _, info in env.summary["node_mapping"]["facilities"].items():
            if info["name"] == "Supplier_SKU3":
                for v in info["units"]["distribution"]["children"]:
                    vehicle_unit = be.world.get_entity_by_id(v["id"])

            if info["name"] == "Warehouse_001":
                dest_facility = be.world.get_facility_by_id(info["id"])
        self.assertTrue(all([vehicle_unit is not None, dest_facility is not None]))

        # there is 80 sku3 in supplier, lets schedule a job for 100, to make sure it will fail to try load
        vehicle_unit.schedule(dest_facility, SKU3_ID, 100, 3)

        # push env to next step
        env.step(None)

        self.assertEqual(100, vehicle_unit.requested_quantity)

        # the patient will -1 as no enough product so load
        self.assertEqual(10 - 1, vehicle_unit._remaining_patient)

        # no payload
        self.assertEqual(0, vehicle_unit.payload)
        self.assertEqual(0, vehicle_unit.data_model.payload)

        # step 9 ticks, patient will be 0
        for i in range(10 - 1):
            env.step(None)

            self.assertEqual(10 - 1 - (i + 1), vehicle_unit._remaining_patient)

        vehicle_nodes = env.snapshot_list["vehicle"]
        features = (
            "id",
            "facility_id",
            "payload",
            "unit_transport_cost"
        )

        states = vehicle_nodes[:vehicle_unit.data_model_index:"payload"].flatten().astype(np.int)

        # no payload from start to now
        self.assertListEqual([0] * 10, list(states))

        # push env to next step, vehicle will be reset to initial state
        env.step(None)

        states = vehicle_nodes[env.frame_index:vehicle_unit.data_model_index:features].flatten().astype(np.int)

        # the product is unloaded, vehicle states will be reset to initial
        # not destination at first
        self.assertIsNone(vehicle_unit._destination)
        self.assertIsNone(vehicle_unit._path)
        self.assertEqual(0, vehicle_unit.product_id)
        self.assertEqual(0, vehicle_unit._remaining_steps)
        self.assertEqual(0, vehicle_unit.payload)
        self.assertEqual(0, vehicle_unit._location)
        self.assertEqual(0, vehicle_unit._velocity)
        self.assertEqual(0, vehicle_unit.requested_quantity)

        # check states

        self.assertEqual(0, states[2])
        self.assertEqual(12, vehicle_unit.data_model.unit_transport_cost)

    def test_vehicle_unit_cannot_unload_at_destination(self):
        """
        NOTE: If vehicle cannot unload at destination, it will keep waiting, until success to unload.

        """
        env = build_env("case_02", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        # try to find first vehicle unit of Supplier
        vehicle_unit: Optional[VehicleUnit] = None
        dest_facility: Optional[FacilityBase] = None

        for _, info in env.summary["node_mapping"]["facilities"].items():
            if info["name"] == "Supplier_SKU3":
                for v in info["units"]["distribution"]["children"]:
                    vehicle_unit = be.world.get_entity_by_id(v["id"])

            if info["name"] == "Warehouse_001":
                dest_facility = be.world.get_facility_by_id(info["id"])
        self.assertTrue(all([vehicle_unit is not None, dest_facility is not None]))

        # move all 80 sku3 to destination, will cause vehicle keep waiting there
        vehicle_unit.schedule(dest_facility, SKU3_ID, 80, 2)

        # step to the end.
        is_done = False

        while not is_done:
            _, _, is_done = env.step(None)

        vehicle_nodes = env.snapshot_list["vehicle"]

        # payload should be 80 for first 4 ticks, as it is on the way
        # then it will unload 100 - 10 - 10 - 10 = 70 products, as this is the remaining space of destination storage
        # so then it will keep waiting to unload remaining 10
        payload_states = vehicle_nodes[:vehicle_unit.data_model_index:"payload"].flatten().astype(np.int)
        self.assertListEqual([80] * 3 + [10] * 97, list(payload_states))

    """
    Distribution unit test:

    . initial state
    . place order
    . dispatch orders without available vehicle
    . dispatch order with vehicle
    """

    def test_distribution_unit_initial_state(self):
        env = build_env("case_02", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        # try to find first vehicle unit of Supplier
        dist_unit: Optional[DistributionUnit] = None
        dest_facility: Optional[FacilityBase] = None

        for _, info in env.summary["node_mapping"]["facilities"].items():
            if info["name"] == "Supplier_SKU3":
                dist_unit = be.world.get_entity_by_id(info["units"]["distribution"]["id"])

            if info["name"] == "Warehouse_001":
                dest_facility = be.world.get_facility_by_id(info["id"])
        self.assertTrue(all([dist_unit is not None, dest_facility is not None]))

        self.assertEqual(0, len(dist_unit._order_queue))

        # reset
        env.reset()

        self.assertEqual(0, len(dist_unit._order_queue))

    def test_distribution_unit_dispatch_order(self):
        env = build_env("case_02", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        # try to find first vehicle unit of Supplier
        dist_unit: Optional[DistributionUnit] = None
        dest_facility: Optional[FacilityBase] = None

        for _, info in env.summary["node_mapping"]["facilities"].items():
            if info["name"] == "Supplier_SKU3":
                dist_unit = be.world.get_entity_by_id(info["units"]["distribution"]["id"])

            if info["name"] == "Warehouse_001":
                dest_facility = be.world.get_facility_by_id(info["id"])
        self.assertTrue(all([dist_unit is not None, dest_facility is not None]))

        first_vehicle: VehicleUnit = dist_unit.vehicles[0]

        order = Order(dest_facility, SKU3_ID, 10, 2)

        dist_unit.place_order(order)

        # check if order is saved
        self.assertEqual(1, len(dist_unit._order_queue))

        # check get pending order correct
        pending_order = dist_unit.get_pending_order()

        self.assertDictEqual({3: 10}, pending_order)

        # same as vehicle schedule case, distribution will try to schedule this order to vehicles from beginning to end
        # so it will dispatch this order to first vehicle
        env.step(None)

        self.assertEqual(dest_facility, first_vehicle._destination)
        self.assertEqual(10, first_vehicle.requested_quantity)
        self.assertEqual(1, first_vehicle._velocity)
        self.assertEqual(SKU3_ID, first_vehicle.product_id)

        # since we already test vehicle unit, do not check the it again here

        # add another order to check pending order
        dist_unit.place_order(order)

        pending_order = dist_unit.get_pending_order()

        self.assertDictEqual({3: 10}, pending_order)

        # another order, will cause the pending order increase
        dist_unit.place_order(order)

        pending_order = dist_unit.get_pending_order()

        # 2 pending orders
        self.assertDictEqual({3: 20}, pending_order)

        # now we have only one available vehicle, 2 pending order
        # next step will cause delay_order_penalty
        env.step(None)

        second_vehicle = dist_unit.vehicles[1]

        self.assertEqual(dest_facility, second_vehicle._destination)
        self.assertEqual(10, second_vehicle.requested_quantity)
        self.assertEqual(1, second_vehicle._velocity)
        self.assertEqual(SKU3_ID, second_vehicle.product_id)

    """
    Seller unit test:
        . initial state
        . with a customized seller unit
        . with built in one
    """

    def test_seller_unit_initial_states(self):
        env = build_env("case_02", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        # find seller for sku3 from retailer facility
        sell_unit: Optional[SellerUnit] = None

        for _, info in env.summary["node_mapping"]["facilities"].items():
            if info["name"] == "Retailer_001":
                for pid, pdetail in info["units"]["products"].items():
                    if pdetail["sku_id"] == SKU3_ID:
                        sell_unit = be.world.get_entity_by_id(pdetail["seller"]["id"])
        self.assertTrue(sell_unit is not None)

        # from configuration
        self.assertEqual(10, sell_unit._gamma)

        self.assertEqual(0, sell_unit._sold)
        self.assertEqual(0, sell_unit._demand)
        self.assertEqual(0, sell_unit._total_sold)
        self.assertEqual(SKU3_ID, sell_unit.product_id)

        #
        self.assertEqual(0, sell_unit.data_model.sold)
        self.assertEqual(0, sell_unit.data_model.demand)
        self.assertEqual(0, sell_unit.data_model.total_sold)
        self.assertEqual(SKU3_ID, sell_unit.product_id)

        env.reset()

        # from configuration
        self.assertEqual(10, sell_unit._gamma)
        self.assertEqual(0, sell_unit._sold)
        self.assertEqual(0, sell_unit._demand)
        self.assertEqual(0, sell_unit._total_sold)
        self.assertEqual(SKU3_ID, sell_unit.product_id)

        #
        self.assertEqual(0, sell_unit.data_model.sold)
        self.assertEqual(0, sell_unit.data_model.demand)
        self.assertEqual(0, sell_unit.data_model.total_sold)
        self.assertEqual(SKU3_ID, sell_unit.product_id)

    def test_seller_unit_demand_states(self):
        env = build_env("case_02", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        # find seller for sku3 from retailer facility
        sell_unit: Optional[SellerUnit] = None

        for _, info in env.summary["node_mapping"]["facilities"].items():
            if info["name"] == "Retailer_001":
                for pid, pdetail in info["units"]["products"].items():
                    if pdetail["sku_id"] == SKU3_ID:
                        sell_unit = be.world.get_entity_by_id(pdetail["seller"]["id"])
        self.assertTrue(sell_unit is not None)

        sku3_init_quantity = sell_unit.facility.skus[SKU3_ID].init_stock

        env.step(None)

        # seller unit will try to count down the product quantity based on demand
        # default seller use gamma distribution on each tick
        demand = sell_unit._demand

        # demand should be same with original
        self.assertEqual(demand, sell_unit.data_model.demand)

        actual_sold = min(demand, sku3_init_quantity)
        # sold may be not same as demand, depend on remaining quantity in storage
        self.assertEqual(actual_sold, sell_unit._sold)
        self.assertEqual(actual_sold, sell_unit.data_model.sold)
        self.assertEqual(actual_sold, sell_unit._total_sold)
        self.assertEqual(actual_sold, sell_unit.data_model.total_sold)

        states = env.snapshot_list["seller"][
                 env.frame_index:sell_unit.data_model_index:("sold", "demand", "total_sold")].flatten().astype(np.int)

        self.assertEqual(actual_sold, states[0])
        self.assertEqual(demand, states[1])
        self.assertEqual(actual_sold, states[2])

        # move to next step to check if state is correct
        env.step(None)

        demand = sell_unit._demand

        # demand should be same with original
        self.assertEqual(demand, sell_unit.data_model.demand)

        actual_sold_2 = min(demand, sku3_init_quantity - actual_sold)

        # sold may be not same as demand, depend on remaining quantity in storage
        self.assertEqual(actual_sold_2, sell_unit._sold)
        self.assertEqual(actual_sold_2, sell_unit.data_model.sold)
        self.assertEqual(actual_sold + actual_sold_2, sell_unit._total_sold)
        self.assertEqual(actual_sold + actual_sold_2, sell_unit.data_model.total_sold)

        states = env.snapshot_list["seller"][
                 env.frame_index:sell_unit.data_model_index:("sold", "demand", "total_sold")].flatten().astype(np.int)

        self.assertEqual(actual_sold_2, states[0])
        self.assertEqual(demand, states[1])
        self.assertEqual(actual_sold + actual_sold_2, states[2])

    def test_seller_unit_customized(self):
        env = build_env("case_03", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        # find seller for sku3 from retailer facility
        sell_unit: Optional[SellerUnit] = None

        for _, info in env.summary["node_mapping"]["facilities"].items():
            if info["name"] == "Retailer_001":
                for pid, pdetail in info["units"]["products"].items():
                    if pdetail["sku_id"] == SKU3_ID:
                        sell_unit = be.world.get_entity_by_id(pdetail["seller"]["id"])
        self.assertTrue(sell_unit is not None)

        # NOTE:
        # this simple seller unit return demands that same as current tick
        env.step(None)

        # so tick 0 will have demand == 0
        # from configuration
        self.assertEqual(0, sell_unit._sold)
        self.assertEqual(0, sell_unit._demand)
        self.assertEqual(0, sell_unit._total_sold)
        self.assertEqual(SKU3_ID, sell_unit.product_id)

        #
        self.assertEqual(0, sell_unit.data_model.sold)
        self.assertEqual(0, sell_unit.data_model.demand)
        self.assertEqual(0, sell_unit.data_model.total_sold)
        self.assertEqual(SKU3_ID, sell_unit.product_id)

        is_done = False

        while not is_done:
            _, _, is_done = env.step(None)

        # check demand history, it should be same as tick
        seller_nodes = env.snapshot_list["seller"]

        demand_states = seller_nodes[:sell_unit.data_model_index:"demand"].flatten().astype(np.int)

        self.assertListEqual([i for i in range(100)], list(demand_states))

        # check sold states
        # it should be 0 after tick 4
        sold_states = seller_nodes[:sell_unit.data_model_index:"sold"].flatten().astype(np.int)
        self.assertListEqual([0, 1, 2, 3, 4] + [0] * 95, list(sold_states))

        # total sold
        total_sold_states = seller_nodes[:sell_unit.data_model_index:"total_sold"].flatten().astype(np.int)
        # total sold will keep same after tick 4
        self.assertListEqual([0, 1, 3, 6, 10] + [10] * 95, list(total_sold_states))


if __name__ == '__main__':
    unittest.main()
