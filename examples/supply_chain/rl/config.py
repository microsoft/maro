# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum
import numpy as np

class VehicleSelection(Enum):
    DEFAULT_ONE = "default"  # Choose the default one
    RANDOM = "random"  # Randomly choosing one for each decision
    SHORTEST_LEADING_TIME = "shortest"  # Always choosing the one with shortest leading time
    CHEAPEST_TOTAL_COST = "cheapest"  # Always choosing the one with cheapest total cost (products, order base, transportation)


distribution_features = ("pending_product_quantity", "pending_order_number")
IDX_DISTRIBUTION_PENDING_PRODUCT_QUANTITY, IDX_DISTRIBUTION_PENDING_ORDER_NUMBER = 0, 1

seller_features = ("total_demand", "sold", "demand")
IDX_SELLER_TOTAL_DEMAND, IDX_SELLER_SOLD, IDX_SELLER_DEMAND = 0, 1, 2

consumer_features = ("order_base_cost", "latest_consumptions")
IDX_CONSUMER_ORDER_BASE_COST, IDX_CONSUMER_LATEST_CONSUMPTIONS = 0, 1

product_features = ("price", )
IDX_PRODUCT_PRICE = 0


vlt_buffer_days = 1.0
num_products_to_sample = 500

ALGO="BSP"
assert ALGO in ["DQN", "EOQ", "PPO", "BSP"], "wrong ALGO"

TEAM_REWARD = False
SHARED_MODEL = False

OR_NUM_CONSUMER_ACTIONS = 20
NUM_CONSUMER_ACTIONS = 3
OR_MANUFACTURE_ACTIONS = 20

# TOPOLOGY = "super_vendor"
TOPOLOGY = f"SCI_{num_products_to_sample}"
# TOPOLOGY = "SCI_1.1"

TRAIN_STEPS = 180
EVAL_STEPS = 60

PLOT_RENDER = True
DUMP_PRODUCT_METRICS = True
LOG_CONSUMER_ACTIONS = True

DUMP_CHOSEN_VLT_INFO = False

env_conf = {
    "scenario": "supply_chain",
    "topology": TOPOLOGY,
    "durations": TRAIN_STEPS,  # number of ticks per episode
}

test_env_conf = {
    "scenario": "supply_chain",
    "topology": TOPOLOGY,
    "durations": TRAIN_STEPS + EVAL_STEPS,  # number of ticks per episode
}

base_policy_conf = {
    "data_loader": "DataLoaderFromHistory",
    "oracle_file": "oracle_samples.csv", # Only need in DataLoaderFromFile loader
    "history_len": 28, # E.g., mapping to np.inf in instance creation if it is static
    "future_len": 7,
    "update_frequency": 3 # E.g., mapping to np.inf in instance creation if no update
}

workflow_settings: dict = {
    "consumption_hist_len": 4,
    "sale_hist_len": 4,
    "pending_order_len": 4,
    "or_policy_vlt_buffer_days": vlt_buffer_days,
    "reward_normalization": 1.0,
    "vehicle_selection_method": VehicleSelection.CHEAPEST_TOTAL_COST,
    "log_path": "examples/supply_chain/logs/",
    "plot_render": PLOT_RENDER,
    "dump_product_metrics": DUMP_PRODUCT_METRICS,
    "log_consumer_actions": LOG_CONSUMER_ACTIONS,
}

EXP_NAME = (
    f"{TOPOLOGY}_{test_env_conf['durations']}_{ALGO}_"
    f"{workflow_settings['vehicle_selection_method']}"
    f"{'_TR' if TEAM_REWARD else ''}"
    f"{'_SM' if SHARED_MODEL else ''}"
)

workflow_settings["log_path"] = f"examples/supply_chain/logs/{EXP_NAME}/"

# 10: 1934612.420211792
# 20: 14355712.158203125
# 50: 9710599.291015625
# 100: 36535436.5625
