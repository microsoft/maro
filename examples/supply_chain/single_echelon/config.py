# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Feature keywords & index for snapshot accessing. NOTE: DO NOT CHANGE.
distribution_features = ("pending_product_quantity", "pending_order_number")
IDX_DISTRIBUTION_PENDING_PRODUCT_QUANTITY, IDX_DISTRIBUTION_PENDING_ORDER_NUMBER = 0, 1

seller_features = ("total_demand", "sold", "demand")
IDX_SELLER_TOTAL_DEMAND, IDX_SELLER_SOLD, IDX_SELLER_DEMAND = 0, 1, 2

consumer_features = ("order_base_cost", "latest_consumptions", "purchased")
IDX_CONSUMER_ORDER_BASE_COST, IDX_CONSUMER_LATEST_CONSUMPTIONS, IDX_CONSUMER_PURCHASED = 0, 1, 2

product_features = ("price",)
IDX_PRODUCT_PRICE = 0

DEVICE: str = "cpu"
# Algorithm for ConsumerUnits: How many products to purchase from the upstream facility?
# EOQ: a consumer baseline policy. The quantity is decided by the expected leading time & the historical demand.
# DQN: a RL policy.
# PPO: a RL policy.
ALGO = "PPO"
assert ALGO in ["DQN", "EOQ", "PPO"], "wrong ALGO"

# Parameters for the reward design of RL policy. Treat ConsumerUnits of one facility as a team or not.
TEAM_REWARD = False
# Parameters for RL policy on SCI topologies only. Let stores in CA, TX and WI to share one model or not.
SHARED_MODEL = True

# Parameters for action shaping (action space discretization).
OR_NUM_CONSUMER_ACTIONS = 20
NUM_CONSUMER_ACTIONS = 3
OR_MANUFACTURE_ACTIONS = 20

# Topology to use, valid SCI topologies:
# - SCI_10_default, SCI_10_cheapest_storage_enlarged, SCI_10_shortest_storage_enlarged
# - SCI_500_default, SCI_500_cheapest_storage_enlarged, SCI_500_shortest_storage_enlarged
# TOPOLOGY = "SCI_10_default"
TOPOLOGY = "single_echelon"

# The duration for training Env. Unit: tick (day).
TRAIN_STEPS = 180
# The extra duration for testing Env. The total duration would be TRAIN_STEPS + EVAL_STEPS. Unit: tick (day).
EVAL_STEPS = 60

# To render figures for agents or not.
# True to enjoy the figures right after the experiment but spend some time.
# Or false to fasten the experiment and get more details with other visualization tool.
PLOT_RENDER = True

env_conf = {
    "scenario": "supply_chain",
    "topology": TOPOLOGY,
    "durations": TRAIN_STEPS,  # Number of ticks per episode
}

test_env_conf = {
    "scenario": "supply_chain",
    "topology": TOPOLOGY,
    "durations": TRAIN_STEPS + EVAL_STEPS,  # Number of ticks per episode
}

workflow_settings: dict = {
    # Parameter for state shaping. How long of consumer features to look back when taking action.
    "consumption_hist_len": 4,
    # Parameter for state shaping. How long of seller features to look back when taking action.
    "sale_hist_len": 4,
    # Parameter for state shaping. How long of pending orders to look back when taking action.
    "pending_order_len": 4,
    # Parameter for reward shaping - reward normalization factor.
    "reward_normalization": 1.0,
    # Render figures for agents or not.
    "plot_render": PLOT_RENDER,
    # Dump product metrics csv to log path or not.
    "dump_product_metrics": True,
    # Dump consumer actions to logger or not.
    "log_consumer_actions": True,
}

# Experiment name, partial setting for log path.
EXP_NAME = (
    f"{TOPOLOGY}"
    # f"_{test_env_conf['durations']}"
    f"_{ALGO}"
    f"{'_TR' if TEAM_REWARD else ''}"
    f"_test"
)

# Path to dump the experimental logs, results, and render figures.
workflow_settings["log_path"] = f"examples/supply_chain/logs/{EXP_NAME}/"
