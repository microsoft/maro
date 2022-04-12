# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

env_conf = {
    "scenario": "supply_chain",
    # Currently available topologies are "sample" or "random". New topologies must consist of a single folder
    # that contains a single config.yml and should be placed under /maro/simulator/scenarios/supply_chain/topologies
    "topology": "plant",
    "durations": 180,  # number of ticks per episode
}

distribution_features = ("pending_product_quantity", "pending_order_number")
seller_features = ("total_demand", "sold", "demand")


OR_NUM_CONSUMER_ACTIONS = 8

ALGO = "EOQ" # PPO or DQN or EOQ
TEAM_REWARD = True

if ALGO in ["PPO"]:
    NUM_CONSUMER_ACTIONS = 3
else:
    NUM_CONSUMER_ACTIONS = 8