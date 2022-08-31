# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from maro.simulator import Env


def build_env(case_name: str, durations: int) -> Env:
    case_folder = os.path.join("tests", "data", "supply_chain", case_name)
    return Env(scenario="supply_chain", topology=case_folder, durations=durations)


def get_product_dict_from_storage(env: Env, frame_index: int, node_index: int) -> Dict[int, int]:
    sku_id_list = env.snapshot_list["storage"][frame_index:node_index:"sku_id_list"].flatten().astype(np.int)
    product_quantity = env.snapshot_list["storage"][frame_index:node_index:"product_quantity"].flatten().astype(np.int)

    return dict(zip(sku_id_list, product_quantity))


def snapshot_query(env: Env, i: int) -> Tuple[
    Dict[int, list], Dict[int, list], Dict[int, list], Dict[int, list], Dict[int, list], Dict[int, list]
]:
    consumer_nodes = env.snapshot_list["consumer"]
    storage_nodes = env.snapshot_list["storage"]
    seller_nodes = env.snapshot_list["seller"]
    manufacture_nodes = env.snapshot_list["manufacture"]
    distribution_nodes = env.snapshot_list["distribution"]

    states_consumer: Dict[int, list] = {}
    states_storage: Dict[int, list] = {}
    states_seller: Dict[int, list] = {}
    states_manufacture: Dict[int, list] = {}
    states_distribution: Dict[int, list] = {}

    env_metric = env.metrics

    for idx in range(len(consumer_nodes)):
        states_consumer[idx] = consumer_nodes[i:idx:consumer_features].flatten().astype(np.float)

    for idx in range(len(storage_nodes)):
        states_storage[idx] = list(storage_nodes[i:idx:storage_features].flatten().astype(np.float))
        states_storage[idx].append(storage_nodes[i:idx:"product_id_list"].flatten().astype(np.int))
        states_storage[idx].append(storage_nodes[i:idx:"product_quantity"].flatten().astype(np.float))
        states_storage[idx].append(storage_nodes[i:idx:"remaining_space"].flatten().astype(np.float))

    for idx in range(len(manufacture_nodes)):
        states_manufacture[idx] = (
            manufacture_nodes[i:idx:manufacture_features]
            .flatten()
            .astype(
                np.float,
            )
        )

    for idx in range(len(distribution_nodes)):
        states_distribution[idx] = (
            distribution_nodes[i:idx:distribution_features]
            .flatten()
            .astype(
                np.float,
            )
        )

    for idx in range(len(seller_nodes)):
        states_seller[idx] = seller_nodes[i:idx:seller_features].flatten().astype(np.float)

    return env_metric, states_consumer, states_storage, states_seller, states_manufacture, states_distribution


def test_env_reset_snapshot_query(
    env: Env,
    action_1: object,
    action_2: object,
    expect_tick: int,
    random_tick: list = None,
) -> List[tuple]:
    snapshots: List[tuple] = []  # List of (env_metric, states_consumer, ..., states_distribution)
    for i in range(expect_tick):
        snapshots.append(snapshot_query(env, i))
        env.step(action_1)

        if random_tick is not None:
            if i in random_tick:
                env.step(action_2)

    return list(zip(*snapshots))


SKU1_ID = 1
SKU2_ID = 2
SKU3_ID = 3
SKU4_ID = 4
FOOD_1_ID = 20
HOBBY_1_ID = 30

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
