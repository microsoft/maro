# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

env_conf = {
    "scenario": "supply_chain",
    "topology": "SCI",
    "durations": 60,  # number of ticks per episode
}

distribution_features = ("pending_product_quantity", "pending_order_number")
IDX_DISTRIBUTION_PENDING_PRODUCT_QUANTITY, IDX_DISTRIBUTION_PENDING_ORDER_NUMBER = 0, 1

seller_features = ("total_demand", "sold", "demand")
IDX_SELLER_TOTAL_DEMAND, IDX_SELLER_SOLD, IDX_SELLER_DEMAND = 0, 1, 2

<<<<<<< HEAD
OR_NUM_CONSUMER_ACTIONS = 10
ALGO="DQN"
TEAM_REWARD = False
SHARED_MODEL = True

if ALGO == "PPO":
    NUM_CONSUMER_ACTIONS = 3
else:
    NUM_CONSUMER_ACTIONS = OR_NUM_CONSUMER_ACTIONS

OR_MANUFACTURE_ACTIONS = 20

=======
consumer_features = ("order_base_cost", "latest_consumptions")
IDX_CONSUMER_ORDER_BASE_COST, IDX_CONSUMER_LATEST_CONSUMPTIONS = 0, 1

NUM_CONSUMER_ACTIONS = 10
>>>>>>> origin/Jinyu/sc_refinement

workflow_settings: dict = {
    "consumption_hist_len": 4,
    "sale_hist_len": 4,
    "pending_order_len": 4,
    # "constraint_state_hist_len": 8,
<<<<<<< HEAD
    "reward_normalization": 1e3,
=======
    "or_policy_vlt_buffer_days": 7,
    "reward_normalization": 1e7,
>>>>>>> origin/Jinyu/sc_refinement
    "default_vehicle_type": "train",
}
