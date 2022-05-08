# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum


class VehicleSelection(Enum):
    FIRST_ONE = 0  # Always choosing the first vehicle type candidate
    RANDOM = 1  # Randomly choosing one for each decision
    SHORTEST_LEADING_TIME = 2  # Always choosing the one with shortest leading time
    CHEAPEST_TOTAL_COST = 3  # Always choosing the one with cheapest total cost (products, order base, transportation)


env_conf = {
    "scenario": "supply_chain",
    "topology": "plant",
    "durations": 100,  # number of ticks per episode
}

distribution_features = ("pending_product_quantity", "pending_order_number")
IDX_DISTRIBUTION_PENDING_PRODUCT_QUANTITY, IDX_DISTRIBUTION_PENDING_ORDER_NUMBER = 0, 1

seller_features = ("total_demand", "sold", "demand")
IDX_SELLER_TOTAL_DEMAND, IDX_SELLER_SOLD, IDX_SELLER_DEMAND = 0, 1, 2

consumer_features = ("order_base_cost", "latest_consumptions")
IDX_CONSUMER_ORDER_BASE_COST, IDX_CONSUMER_LATEST_CONSUMPTIONS = 0, 1

NUM_CONSUMER_ACTIONS = 10

workflow_settings: dict = {
    "consumption_hist_len": 4,
    "sale_hist_len": 4,
    "pending_order_len": 4,
    # "constraint_state_hist_len": 8,
    "or_policy_vlt_buffer_days": 7,
    "reward_normalization": 1e7,
    "default_vehicle_type": None,
    "vehicle_selection_method": VehicleSelection.FIRST_ONE,
}
