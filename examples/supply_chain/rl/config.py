# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

env_conf = {
    "scenario": "supply_chain",
    "topology": "SCI",
    "durations": 180,  # number of ticks per episode
}

EVAL_STEPS = 60

test_env_conf = {
    "scenario": "supply_chain",
    "topology": "SCI",
    "durations": 180+EVAL_STEPS,  # number of ticks per episode
}


distribution_features = ("pending_product_quantity", "pending_order_number")
IDX_DISTRIBUTION_PENDING_PRODUCT_QUANTITY, IDX_DISTRIBUTION_PENDING_ORDER_NUMBER = 0, 1

seller_features = ("total_demand", "sold", "demand")
IDX_SELLER_TOTAL_DEMAND, IDX_SELLER_SOLD, IDX_SELLER_DEMAND = 0, 1, 2

consumer_features = ("order_base_cost", "latest_consumptions")
IDX_CONSUMER_ORDER_BASE_COST, IDX_CONSUMER_LATEST_CONSUMPTIONS = 0, 1


OR_NUM_CONSUMER_ACTIONS = 10

# ALGO="EOQ"
# EXP_NAME = "BASELINE_SCI_50SKUs_DIST"
# 10: 1934612.420211792
# 50: 9710599.291015625

ALGO="DQN"
EXP_NAME = "SCI_50SKUs_DIST_DQN"

# ALGO="PPO"
# EXP_NAME = "SCI_10SKUs_DIST_PPO"

assert ALGO in ["DQN", "EOQ", "PPO"], "wrong ALGO"
TEAM_REWARD = False
SHARED_MODEL = True

NUM_CONSUMER_ACTIONS = 3
# if ALGO == "PPO":
#     NUM_CONSUMER_ACTIONS = 3
# else:
#     NUM_CONSUMER_ACTIONS = OR_NUM_CONSUMER_ACTIONS

OR_MANUFACTURE_ACTIONS = 20


workflow_settings: dict = {
    "consumption_hist_len": 4,
    "sale_hist_len": 4,
    "pending_order_len": 4,
    "reward_normalization": 1.0,
    "or_policy_vlt_buffer_days": 1.5,
    "default_vehicle_type": "train",
}
