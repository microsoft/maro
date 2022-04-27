# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from os.path import dirname, join, realpath
import pandas as pd

from maro.rl.workflows.scenario import Scenario
from examples.supply_chain.rl.render_tools import SimulationTracker, SimulationTrackerHtml

# from examples.supply_chain.rl.policies import agent2baseline_policy

# config variables
SCENARIO_NAME = "supply_chain"
SCENARIO_PATH = os.path.join("examples", SCENARIO_NAME, "rl")
import sys
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default="Round1")
    args = parser.parse_args()
    scenario = Scenario(SCENARIO_PATH)
    policy_creator = scenario.policy_creator
    policy_dict = {name: get_policy_func(name) for name, get_policy_func in policy_creator.items()}
    policy_creator = {name: lambda name: policy_dict[name] for name in policy_dict}

    env_sampler = scenario.env_sampler_creator(policy_creator)

    result = env_sampler.eval()
    if scenario.post_evaluate:
        scenario.post_evaluate(result["info"], 0)

    LOG_PATH = join(dirname(SCENARIO_PATH), "results", f"baseline_{args.exp_name}")
    os.makedirs(LOG_PATH, exist_ok=True)
    # facility_types = ['ProductUnit', 'StoreProductUnit']
    tracker = result['tracker']
    tracker.render('%s/a_plot_balance.png' %
                    LOG_PATH, tracker.step_balances, ["OuterRetailerFacility"])
    tracker.render('%s/a_plot_reward.png' %
                LOG_PATH, tracker.step_rewards, ["OuterRetailerFacility"])
    tracker.render_sku(LOG_PATH)
    
    df_product = pd.DataFrame(env_sampler._balance_calculator.product_metric_track)
    print(df_product.head())
    # df_product = df_product.groupby(['tick', 'id']).first().reset_index()
    result_file_loc = f'{LOG_PATH}/output_product_metrics.csv'
    df_product.to_csv(result_file_loc, index=False)
    # html_render = SimulationTrackerHtml(result_file_loc)
    # html_render.render_sku()
    # html_render.render_facility()
    mean_reward_path = join(dirname(SCENARIO_PATH), "rl")
    with open(f"{mean_reward_path}/baseline_mean_reward.py", "w") as f:
        f.write("mean_reward = " + str(result["mean_reward"]))