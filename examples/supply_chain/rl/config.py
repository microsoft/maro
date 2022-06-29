# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum

import numpy as np

from maro.simulator.scenarios.supply_chain.facilities import FacilityInfo, OuterRetailerFacility
from maro.simulator.scenarios.supply_chain.objects import SupplyChainEntity
from maro.simulator.scenarios.supply_chain.units import ConsumerUnit, ManufactureUnit


class VehicleSelection(Enum):
    DEFAULT_ONE = "default"  # Choose the default one
    RANDOM = "random"  # Randomly choosing one for each decision
    SHORTEST_LEADING_TIME = "shortest"  # Always choosing the one with shortest leading time
    CHEAPEST_TOTAL_COST = (
        "cheapest"  # Always choosing the one with cheapest total cost (products, order base, transportation)
    )


distribution_features = ("pending_product_quantity", "pending_order_number")
IDX_DISTRIBUTION_PENDING_PRODUCT_QUANTITY, IDX_DISTRIBUTION_PENDING_ORDER_NUMBER = 0, 1

seller_features = ("total_demand", "sold", "demand")
IDX_SELLER_TOTAL_DEMAND, IDX_SELLER_SOLD, IDX_SELLER_DEMAND = 0, 1, 2

consumer_features = ("order_base_cost", "latest_consumptions", "purchased")
IDX_CONSUMER_ORDER_BASE_COST, IDX_CONSUMER_LATEST_CONSUMPTIONS, IDX_CONSUMER_PURCHASED = 0, 1, 2

product_features = ("price",)
IDX_PRODUCT_PRICE = 0


m_vlt, s_vlt, ns_vlt = 2, 2, 2


def get_vlt_buffer_factor(entity: SupplyChainEntity, facility_info: FacilityInfo) -> float:
    if issubclass(entity.class_type, ManufactureUnit):
        return m_vlt
    elif issubclass(entity.class_type, ConsumerUnit):
        if issubclass(facility_info.class_name, OuterRetailerFacility):
            return s_vlt
        else:
            return ns_vlt
    else:
        raise (f"Get entity(id: {entity.id}) neither ManufactureUnit nor ConsumerUnit")


ALGO = "BSP"
assert ALGO in ["DQN", "EOQ", "PPO", "BSP"], "wrong ALGO"

TEAM_REWARD = False
SHARED_MODEL = False

OR_NUM_CONSUMER_ACTIONS = 20
NUM_CONSUMER_ACTIONS = 3
OR_MANUFACTURE_ACTIONS = 20

num_products_to_sample = 500
selection = VehicleSelection.SHORTEST_LEADING_TIME
storage_enlarged = False

# TOPOLOGY = (
#     f"SCI_{num_products_to_sample}"
#     f"_{selection.value}"
#     f"{'_storage_enlarged' if storage_enlarged else ''}"
# )
TOPOLOGY = "walmart_3_layers"

TRAIN_STEPS = 1
EVAL_STEPS = 91

PLOT_RENDER = False

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

base_policy_conf = {
    "data_loader": "OracleDataLoader",

    # Oracle file only need in OracleDataLoader
    "oracle_file_dir": "maro/simulator/scenarios/supply_chain/topologies/walmart_3_layers/data",
    "oracle_files": [
        "Store_4803.csv", "Store_6649.csv", "Store_6685.csv", "Store_6688.csv", "Store_6743.csv", 
        "Store_6773.csv", "Store_4830.csv", "Store_6107.csv", "Store_6686.csv", "Store_6687.csv", 
        "Store_6752.csv", "Store_6765.csv", "Store_6505.csv", "Store_6640.csv", "Store_6648.csv",
        "Store_6672.csv", "Store_6673.csv", "Store_6684.csv", "Store_6753.csv", "Store_6822.csv"
    ],
    "history_len": np.inf,  # E.g., mapping to np.inf in instance creation if it is static
    "future_len": np.inf,
    "update_frequency": np.inf,  # E.g., mapping to np.inf in instance creation if no update

    # If true, until next update, all steps will share the same stock level
    # otherwise, each steps will calculate own stock level.
    "share_same_stock_level": False,
    "start_date_time": "2021-07-01",
    "durations": TRAIN_STEPS + EVAL_STEPS,
}

workflow_settings: dict = {
    "consumption_hist_len": 4,
    "sale_hist_len": 4,
    "pending_order_len": 4,
    "reward_normalization": 1.0,
    "vehicle_selection_method": VehicleSelection.CHEAPEST_TOTAL_COST,
    "log_path": "examples/supply_chain/logs/",
    "plot_render": PLOT_RENDER,
    "dump_product_metrics": True,
    "log_consumer_actions": True,
    "dump_chosen_vlt_info": True,
}

EXP_NAME = (
    f"{TOPOLOGY}"
    # f"_{test_env_conf['durations']}"
    # f"_{workflow_settings['vehicle_selection_method'].value}"
    f"_{ALGO}"
    f"{'_TR' if TEAM_REWARD else ''}"
    f"{'_SM' if SHARED_MODEL else ''}"
    # f"_vlt-{m_vlt}-{s_vlt}-{ns_vlt}"
)

workflow_settings["log_path"] = f"examples/supply_chain/logs/{EXP_NAME}/"
