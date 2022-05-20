# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum


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


vlt_buffer_days = 1
num_products_to_sample = 10

ALGO="EOQ"
assert ALGO in ["DQN", "EOQ", "PPO"], "wrong ALGO"

TEAM_REWARD = False
SHARED_MODEL = False

OR_NUM_CONSUMER_ACTIONS = 20
NUM_CONSUMER_ACTIONS = 3
OR_MANUFACTURE_ACTIONS = 20

TOPOLOGY = f"SCI_{num_products_to_sample}_default"
# TOPOLOGY = f"SCI_{num_products_to_sample}_shortest_no_ring"
# TOPOLOGY = f"SCI_{num_products_to_sample}_cheapest_no_ring"

TRAIN_STEPS = 180
EVAL_STEPS = 60

PLOT_RENDER = False

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

workflow_settings: dict = {
    "consumption_hist_len": 4,
    "sale_hist_len": 4,
    "pending_order_len": 4,
    "or_policy_vlt_buffer_days": vlt_buffer_days,
    "reward_normalization": 1.0,
    "vehicle_selection_method": VehicleSelection.CHEAPEST_TOTAL_COST,
    "log_path": "examples/supply_chain/logs/",
    "plot_render": PLOT_RENDER,
    "dump_product_metrics": True,
    "log_consumer_actions": True,
    "dump_chosen_vlt_info": True,
}

EXP_NAME = (
    f"{TOPOLOGY}_{test_env_conf['durations']}_{ALGO}"
    f"{'_TR' if TEAM_REWARD else ''}"
    f"{'_SM' if SHARED_MODEL else ''}"
)

workflow_settings["log_path"] = f"examples/supply_chain/logs/{EXP_NAME}/"
