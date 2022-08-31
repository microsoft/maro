# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
import unittest
from typing import List

import numpy as np

from maro.simulator.scenarios.supply_chain import (
    ConsumerAction,
    ConsumerUnit,
    FacilityBase,
    ManufactureAction,
    ManufactureUnit,
    RetailerFacility,
)
from maro.simulator.scenarios.supply_chain.business_engine import SupplyChainBusinessEngine

from tests.supply_chain.common import (
    SKU1_ID,
    SKU3_ID,
    build_env,
    get_product_dict_from_storage,
    snapshot_query,
    test_env_reset_snapshot_query,
)


class MyTestCase(unittest.TestCase):
    """
    . test env reset with none action
    . with ManufactureAction only
    . with ConsumerAction only
    . with both ManufactureAction and ConsumerAction
    """

    def test_env_reset_with_none_action(self) -> None:
        """test_env_reset_with_none_action"""
        env = build_env("case_05", 500)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        env.step(None)

        # ##################################### Before reset #####################################

        expect_tick = 10

        # Save the env.metric of each tick into env_metric_1
        # Store the information about the snapshot unit of each tick in states_1_unit
        (
            env_metric_1,
            states_1_consumer,
            states_1_storage,
            states_1_seller,
            states_1_manufacture,
            states_1_distribution,
        ) = test_env_reset_snapshot_query(
            env=env,
            action_1=None,
            action_2=None,
            expect_tick=expect_tick,
            random_tick=None,
        )

        # ############################### Test whether reset updates the storage unit completely ################
        env.reset()
        env.step(None)

        # Check snapshot initial state after env.reset()
        (
            env_metric_initial,
            states_consumer_initial,
            states_storage_initial,
            states_seller_initial,
            states_manufacture_initial,
            states_distribution_initial,
        ) = snapshot_query(env, 0)
        self.assertListEqual(list(states_1_consumer[0]), list(states_consumer_initial))
        self.assertListEqual(list(states_1_storage[0]), list(states_storage_initial))
        self.assertListEqual(list(states_1_seller[0]), list(states_seller_initial))
        self.assertListEqual(list(states_1_manufacture[0]), list(states_manufacture_initial))
        self.assertListEqual(list(states_1_distribution[0]), list(states_distribution_initial))
        self.assertListEqual(list(env_metric_1[0].values()), list(env_metric_initial.values()))

        # Save the env.metric of each tick into env_metric_2
        # Store the information about the snapshot unit of each tick in states_2_unit
        (
            env_metric_2,
            states_2_consumer,
            states_2_storage,
            states_2_seller,
            states_2_manufacture,
            states_2_distribution,
        ) = test_env_reset_snapshot_query(
            env=env,
            action_1=None,
            action_2=None,
            expect_tick=expect_tick,
            random_tick=None,
        )

        for i in range(expect_tick):
            self.assertListEqual(list(states_1_consumer[i]), list(states_2_consumer[i]))
            self.assertListEqual(list(states_1_storage[i]), list(states_2_storage[i]))
            self.assertListEqual(list(states_1_seller[i]), list(states_2_seller[i]))
            self.assertListEqual(list(states_1_manufacture[i]), list(states_2_manufacture[i]))
            self.assertListEqual(list(states_1_distribution[i]), list(states_2_distribution[i]))
            self.assertListEqual(list(env_metric_1[i].values()), list(env_metric_2[i].values()))

    def test_env_reset_with_ManufactureAction_only(self) -> None:
        """test env reset with ManufactureAction only"""
        env = build_env("case_02", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        supplier_3: FacilityBase = be.world._get_facility_by_name("Supplier_SKU3")

        storage_unit = supplier_3.storage
        manufacture_unit = supplier_3.products[SKU3_ID].manufacture
        storage_nodes = env.snapshot_list["storage"]

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

        expect_tick = 30

        action_1 = ManufactureAction(manufacture_unit.id, 1)
        action_2 = ManufactureAction(manufacture_unit.id, 0)

        random_tick: List[int] = []

        # The purpose is to randomly perform the order operation
        for i in range(10):
            random_tick.append(random.randint(1, 30))

            # Save the env.metric of each tick into env_metric_1
            # Store the information about the snapshot unit of each tick in states_1_unit
            (
                env_metric_1,
                states_1_consumer,
                states_1_storage,
                states_1_seller,
                states_1_manufacture,
                states_1_distribution,
            ) = test_env_reset_snapshot_query(
                env=env,
                action_1=action_1,
                action_2=action_2,
                expect_tick=expect_tick,
                random_tick=random_tick,
            )

        # ############################### Test whether reset updates the manufacture unit completely ################
        env.reset()
        env.step(None)

        # Check snapshot initial state after env.reset()
        (
            env_metric_initial,
            states_consumer_initial,
            states_storage_initial,
            states_seller_initial,
            states_manufacture_initial,
            states_distribution_initial,
        ) = snapshot_query(env, 0)
        self.assertListEqual(list(states_1_consumer[0]), list(states_consumer_initial))
        self.assertListEqual(list(states_1_storage[0]), list(states_storage_initial))
        self.assertListEqual(list(states_1_seller[0]), list(states_seller_initial))
        self.assertListEqual(list(states_1_manufacture[0]), list(states_manufacture_initial))
        self.assertListEqual(list(states_1_distribution[0]), list(states_distribution_initial))
        self.assertListEqual(list(env_metric_1[0].values()), list(env_metric_initial.values()))

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
        # Store the information about the snapshot unit of each tick in states_2_unit
        (
            env_metric_2,
            states_2_consumer,
            states_2_storage,
            states_2_seller,
            states_2_manufacture,
            states_2_distribution,
        ) = test_env_reset_snapshot_query(
            env=env,
            action_1=action_1,
            action_2=action_2,
            expect_tick=expect_tick,
            random_tick=random_tick,
        )

        for i in range(expect_tick):
            self.assertListEqual(list(states_1_consumer[i]), list(states_2_consumer[i]))
            self.assertListEqual(list(states_1_storage[i]), list(states_2_storage[i]))
            self.assertListEqual(list(states_1_seller[i]), list(states_2_seller[i]))
            self.assertListEqual(list(states_1_manufacture[i]), list(states_2_manufacture[i]))
            self.assertListEqual(list(states_1_distribution[i]), list(states_2_distribution[i]))
            self.assertListEqual(list(env_metric_1[i].values()), list(env_metric_2[i].values()))

    def test_env_reset_with_ConsumerAction_only(self) -> None:
        """ "test env reset with ConsumerAction only"""
        env = build_env("case_05", 500)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        env.step(None)

        warehouse_1 = be.world._get_facility_by_name("Warehouse_001")
        supplier_3: FacilityBase = be.world._get_facility_by_name("Supplier_SKU3")
        consumer_unit = warehouse_1.products[SKU3_ID].consumer

        # ##################################### Before reset #####################################
        action = ConsumerAction(consumer_unit.id, SKU3_ID, supplier_3.id, 1, "train")
        expect_tick = 100

        # Save the env.metric of each tick into env_metric_1
        # Store the information about the snapshot unit of each tick in states_1_unit
        (
            env_metric_1,
            states_1_consumer,
            states_1_storage,
            states_1_seller,
            states_1_manufacture,
            states_1_distribution,
        ) = test_env_reset_snapshot_query(
            env=env,
            action_1=action,
            action_2=None,
            expect_tick=expect_tick,
            random_tick=None,
        )

        # ############### Test whether reset updates the consumer unit completely ################
        env.reset()
        env.step(None)

        # Check snapshot initial state after env.reset()
        (
            env_metric_initial,
            states_consumer_initial,
            states_storage_initial,
            states_seller_initial,
            states_manufacture_initial,
            states_distribution_initial,
        ) = snapshot_query(env, 0)
        self.assertListEqual(list(states_1_consumer[0]), list(states_consumer_initial))
        self.assertListEqual(list(states_1_storage[0]), list(states_storage_initial))
        self.assertListEqual(list(states_1_seller[0]), list(states_seller_initial))
        self.assertListEqual(list(states_1_manufacture[0]), list(states_manufacture_initial))
        self.assertListEqual(list(states_1_distribution[0]), list(states_distribution_initial))
        self.assertListEqual(list(env_metric_1[0].values()), list(env_metric_initial.values()))

        # Save the env.metric of each tick into env_metric_2
        # Store the information about the snapshot unit of each tick in states_2_unit
        (
            env_metric_2,
            states_2_consumer,
            states_2_storage,
            states_2_seller,
            states_2_manufacture,
            states_2_distribution,
        ) = test_env_reset_snapshot_query(
            env=env,
            action_1=action,
            action_2=None,
            expect_tick=expect_tick,
            random_tick=None,
        )

        for i in range(expect_tick):
            self.assertListEqual(list(states_1_consumer[i]), list(states_2_consumer[i]))
            self.assertListEqual(list(states_1_storage[i]), list(states_2_storage[i]))
            self.assertListEqual(list(states_1_seller[i]), list(states_2_seller[i]))
            self.assertListEqual(list(states_1_manufacture[i]), list(states_2_manufacture[i]))
            self.assertListEqual(list(states_1_distribution[i]), list(states_2_distribution[i]))
            self.assertListEqual(list(env_metric_1[i].values()), list(env_metric_2[i].values()))

    def test_env_reset_with_both_ManufactureAction_and_ConsumerAction(self) -> None:
        """test env reset with both ManufactureAction and ConsumerAction"""
        env = build_env("case_05", 100)
        be = env.business_engine
        assert isinstance(be, SupplyChainBusinessEngine)

        env.step(None)

        supplier_1: FacilityBase = be.world._get_facility_by_name("Supplier_SKU1")
        warehouse_1: RetailerFacility = be.world._get_facility_by_name("Warehouse_001")
        consumer_unit: ConsumerUnit = warehouse_1.products[SKU1_ID].consumer
        manufacture_unit: ManufactureUnit = supplier_1.products[SKU1_ID].manufacture

        # ##################################### Before reset #####################################
        action_consumer = ConsumerAction(consumer_unit.id, SKU1_ID, supplier_1.id, 5, "train")
        action_manufacture = ManufactureAction(manufacture_unit.id, 1)

        expect_tick = 100

        random_tick: List[int] = []

        # The purpose is to randomly perform the order operation
        for i in range(30):
            random_tick.append(random.randint(0, 90))

        # Save the env.metric of each tick into env_metric_1
        # Store the information about the snapshot unit of each tick in states_1_unit
        (
            env_metric_1,
            states_1_consumer,
            states_1_storage,
            states_1_seller,
            states_1_manufacture,
            states_1_distribution,
        ) = test_env_reset_snapshot_query(
            env=env,
            action_1=action_consumer,
            action_2=action_manufacture,
            expect_tick=expect_tick,
            random_tick=random_tick,
        )

        # ############### Test whether reset updates the consumer unit completely ################
        env.reset()
        env.step(None)

        # Check snapshot initial state after env.reset()
        (
            env_metric_initial,
            states_consumer_initial,
            states_storage_initial,
            states_seller_initial,
            states_manufacture_initial,
            states_distribution_initial,
        ) = snapshot_query(env, 0)
        self.assertListEqual(list(states_1_consumer[0]), list(states_consumer_initial))
        self.assertListEqual(list(states_1_storage[0]), list(states_storage_initial))
        self.assertListEqual(list(states_1_seller[0]), list(states_seller_initial))
        self.assertListEqual(list(states_1_manufacture[0]), list(states_manufacture_initial))
        self.assertListEqual(list(states_1_distribution[0]), list(states_distribution_initial))
        self.assertListEqual(list(env_metric_1[0].values()), list(env_metric_initial.values()))

        # Save the env.metric of each tick into env_metric_2
        # Store the information about the snapshot unit of each tick in states_2_unit
        (
            env_metric_2,
            states_2_consumer,
            states_2_storage,
            states_2_seller,
            states_2_manufacture,
            states_2_distribution,
        ) = test_env_reset_snapshot_query(
            env=env,
            action_1=action_consumer,
            action_2=action_manufacture,
            expect_tick=expect_tick,
            random_tick=random_tick,
        )

        for i in range(expect_tick):
            self.assertListEqual(list(states_1_consumer[i]), list(states_2_consumer[i]))
            self.assertListEqual(list(states_1_storage[i]), list(states_2_storage[i]))
            self.assertListEqual(list(states_1_seller[i]), list(states_2_seller[i]))
            self.assertListEqual(list(states_1_manufacture[i]), list(states_2_manufacture[i]))
            self.assertListEqual(list(states_1_distribution[i]), list(states_2_distribution[i]))
            self.assertListEqual(list(env_metric_1[i].values()), list(env_metric_2[i].values()))


if __name__ == "__main__":
    unittest.main()
