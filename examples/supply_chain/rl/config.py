# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

distribution_features = ("pending_product_quantity", "pending_order_number")
IDX_DISTRIBUTION_PENDING_PRODUCT_QUANTITY, IDX_DISTRIBUTION_PENDING_ORDER_NUMBER = 0, 1

seller_features = ("total_demand", "sold", "demand")
IDX_SELLER_TOTAL_DEMAND, IDX_SELLER_SOLD, IDX_SELLER_DEMAND = 0, 1, 2

consumer_features = ("order_base_cost", "latest_consumptions")
IDX_CONSUMER_ORDER_BASE_COST, IDX_CONSUMER_LATEST_CONSUMPTIONS = 0, 1



vlt_buffer_days = 1.0
num_products_to_sample = 10

ALGO="EOQ"

TEAM_REWARD = False
SHARED_MODEL = False

assert ALGO in ["DQN", "EOQ", "PPO"], "wrong ALGO"
OR_NUM_CONSUMER_ACTIONS = 20
NUM_CONSUMER_ACTIONS = 3
# if ALGO == "PPO":
#     NUM_CONSUMER_ACTIONS = 3
# else:
#     NUM_CONSUMER_ACTIONS = OR_NUM_CONSUMER_ACTIONS
OR_MANUFACTURE_ACTIONS = 20

TRAIN_STEPS = 1
EVAL_STEPS = 2


env_conf = {
    "scenario": "supply_chain",
    "topology": f"SCI_{num_products_to_sample}",
    # "topology": "SCI_1.1",
    # "topology": "super_vendor",
    "durations": TRAIN_STEPS,  # number of ticks per episode
}


test_env_conf = {
    "scenario": "supply_chain",
    "topology": f"SCI_{num_products_to_sample}",
    # "topology": "SCI_1.1",
    # "topology": "super_vendor",
    "durations": TRAIN_STEPS+EVAL_STEPS,  # number of ticks per episode
}


from enum import Enum


class VehicleSelection(Enum):
    FIRST_ONE = 0  # Always choosing the first vehicle type candidate
    RANDOM = 1  # Randomly choosing one for each decision
    SHORTEST_LEADING_TIME = 2  # Always choosing the one with shortest leading time
    CHEAPEST_TOTAL_COST = 3  # Always choosing the one with cheapest total cost (products, order base, transportation)


workflow_settings: dict = {
    "consumption_hist_len": 4,
    "sale_hist_len": 4,
    "pending_order_len": 4,
    "reward_normalization": 1.0,
    "or_policy_vlt_buffer_days": vlt_buffer_days,
    "default_vehicle_type": None,
    "vehicle_selection_method": VehicleSelection.CHEAPEST_TOTAL_COST,
}

# EXP_NAME = f"{env_conf['topology']}_{ALGO}_{workflow_settings['vehicle_selection_method']}"
EXP_NAME = f"sc_render_{test_env_conf['topology']}"

# EXP_NAME = f"{ALGO}_SCI_{num_products_to_sample}SKUs_DIST_{vlt_buffer_days}"
# if TEAM_REWARD:
#     EXP_NAME += '_TR'
# if SHARED_MODEL:
#     EXP_NAME += "_SM"
# 10: 1934612.420211792
# 20: 14355712.158203125
# 50: 9710599.291015625
# 100: 36535436.5625
