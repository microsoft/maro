# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

distribution_features = ("pending_product_quantity", "pending_order_number")
IDX_DISTRIBUTION_PENDING_PRODUCT_QUANTITY, IDX_DISTRIBUTION_PENDING_ORDER_NUMBER = 0, 1

seller_features = ("total_demand", "sold", "demand")
IDX_SELLER_TOTAL_DEMAND, IDX_SELLER_SOLD, IDX_SELLER_DEMAND = 0, 1, 2

consumer_features = ("order_base_cost", "latest_consumptions")
IDX_CONSUMER_ORDER_BASE_COST, IDX_CONSUMER_LATEST_CONSUMPTIONS = 0, 1



vlt_buffer_days = 1.2
num_products_to_sample = 1000

ALGO="EOQ"

TEAM_REWARD = False
SHARED_MODEL = False

EXP_NAME = f"{ALGO}_SCI_{num_products_to_sample}SKUs_DIST_{vlt_buffer_days}"
if TEAM_REWARD:
    EXP_NAME += '_TR'
if SHARED_MODEL:
    EXP_NAME += "_SM"
# 10: 1934612.420211792
# 20: 14355712.158203125
# 50: 9710599.291015625
# 100: 36535436.5625


assert ALGO in ["DQN", "EOQ", "PPO"], "wrong ALGO"
OR_NUM_CONSUMER_ACTIONS = 20
NUM_CONSUMER_ACTIONS = 3
# if ALGO == "PPO":
#     NUM_CONSUMER_ACTIONS = 3
# else:
#     NUM_CONSUMER_ACTIONS = OR_NUM_CONSUMER_ACTIONS
OR_MANUFACTURE_ACTIONS = 20



env_conf = {
    "scenario": "supply_chain",
    "topology": f"SCI_{num_products_to_sample}",
    "durations": 180,  # number of ticks per episode
}

EVAL_STEPS = 60

test_env_conf = {
    "scenario": "supply_chain",
    "topology": f"SCI_{num_products_to_sample}",
    "durations": 180+EVAL_STEPS,  # number of ticks per episode
}


workflow_settings: dict = {
    "consumption_hist_len": 4,
    "sale_hist_len": 4,
    "pending_order_len": 4,
    "reward_normalization": 1.0,
    "or_policy_vlt_buffer_days": vlt_buffer_days,
    "default_vehicle_type": "train",
}