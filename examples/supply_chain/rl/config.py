# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

env_conf = {
    "scenario": "supply_chain",
    # Currently available topologies are "sample" or "random". New topologies must consist of a single folder
    # that contains a single config.yml and should be placed under /maro/simulator/scenarios/supply_chain/topologies
    "topology": "walmart",
    "durations": 180,  # number of ticks per episode
}

distribution_features = ("pending_product_quantity", "pending_order_number")
seller_features = ("total_demand", "sold", "demand")

NUM_CONSUMER_ACTIONS = 4
OR_NUM_CONSUMER_ACTIONS = 8

ALGO = "DQN" # PPO or DQN
TEAM_REWARD = True
