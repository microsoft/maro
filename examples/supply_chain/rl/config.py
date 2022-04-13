# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

env_conf = {
    "scenario": "supply_chain",
    # Currently available topologies are "sample" or "random". New topologies must consist of a single folder
    # that contains a single config.yml and should be placed under /maro/simulator/scenarios/supply_chain/topologies
    "topology": "plant",
    "durations": 100,  # number of ticks per episode
}

distribution_features = ("pending_product_quantity", "pending_order_number")
seller_features = ("total_demand", "sold", "demand")

NUM_CONSUMER_ACTIONS = 10

workflow_settings: dict = {
    "global_reward_weight_producer": 0.50,
    "global_reward_weight_consumer": 0.50,
    "downsampling_rate": 1,
    "episode_duration": 21,
    "initial_balance": 100000,
    "consumption_hist_len": 4,
    "sale_hist_len": 4,
    "pending_order_len": 4,
    "constraint_state_hist_len": 8,
    "total_echelons": 3,
    "replenishment_discount": 0.9,
    "reward_normalization": 1e7,
    "constraint_violate_reward": -1e6,
    "gamma": 0.99,
    "tail_timesteps": 7,
    "heading_timesteps": 7,
}
