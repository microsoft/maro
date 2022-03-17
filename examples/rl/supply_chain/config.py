# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from os.path import dirname, join, realpath

env_conf = {
    "scenario": "supply_chain",
    # Currently available topologies are "sample" or "random". New topologies must consist of a single folder
    # that contains a single config.yml and should be placed under /maro/simulator/scenarios/supply_chain/topologies
    "topology": "sample",
    "durations": 100,  # number of ticks per episode
}

distribution_features = ("remaining_order_quantity", "remaining_order_number")
seller_features = ("total_demand", "sold", "demand")

rl_algorithm = "null"

NUM_CONSUMER_ACTIONS = 10
