# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import sys
sys.path.append("/data/songlei/maro/")
from os.path import dirname, join, realpath

from maro.rl.training import TrainingManager
from maro.rl.workflows.scenario import Scenario
from maro.utils import LoggerV2

# config variables
SCENARIO_NAME = "supply_chain"
SCENARIO_PATH = join(dirname(dirname(realpath(__file__))), SCENARIO_NAME, "rl")
NUM_EPISODES = 1000
NUM_STEPS = 4
CHECKPOINT_PATH = join(dirname(SCENARIO_PATH), "checkpoints")
CHECKPOINT_INTERVAL = 10
EVAL_SCHEDULE = list(range(100, NUM_EPISODES+50, 50))


import argparse
import wandb
import os
os.environ["WANDB_API_KEY"] = "116a4f287fd4fbaa6f790a50d2dd7f97ceae4a03"
wandb.login()
import pandas as pd

# Single-threaded launcher
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default="Round1")
    parser.add_argument("--baseline", action='store_true')
    parser.add_argument("--team_reward", action='store_true')
    parser.add_argument("--shared_model", action='store_true')
    args = parser.parse_args()
    
    LOG_PATH = join(dirname(SCENARIO_PATH), "results", args.exp_name)
    os.makedirs(LOG_PATH, exist_ok=True)

    scenario = Scenario(SCENARIO_PATH)
    logger = LoggerV2("MAIN", dump_path=f"{LOG_PATH}/log.txt")

    agent2policy = scenario.agent2policy
    policy_creator = scenario.policy_creator
    policy_dict = {name: get_policy_func(name) for name, get_policy_func in policy_creator.items()}
    policy_creator = {name: lambda name: policy_dict[name] for name in policy_dict}
    trainer_creator = scenario.trainer_creator

    # evaluation schedule
    logger.info(f"Policy will be evaluated at the end of episodes {EVAL_SCHEDULE}")
    eval_point_index = 0

    if scenario.trainable_policies is None:
        trainable_policies = set(policy_creator.keys())
    else:
        trainable_policies = set(scenario.trainable_policies)

    env_sampler = scenario.env_sampler_creator(policy_creator)

    trainable_policy_creator = {name: func for name, func in policy_creator.items() if name in trainable_policies}
    trainable_agent2policy = {id_: name for id_, name in agent2policy.items() if name in trainable_policies}
    training_manager = TrainingManager(
        trainable_policy_creator,
        trainer_creator,
        trainable_agent2policy,
        device_mapping=scenario.device_mapping,
        logger=logger
    )

    # main loopxs
    for ep in range(1, NUM_EPISODES + 1):
        collect_time = training_time = 0
        segment, end_of_episode = 1, False
        while not end_of_episode:
            # experience collection
            result = env_sampler.sample(num_steps=NUM_STEPS)
            experiences = result["experiences"]
            end_of_episode = result["end_of_episode"]
            if scenario.post_collect:
                scenario.post_collect(result["info"], ep, segment)
            logger.info(f"Roll-out completed for episode {ep}. Training started...")
            training_manager.record_experiences(experiences)
            training_manager.train_step()
            segment += 1
            
        if CHECKPOINT_PATH and ep % CHECKPOINT_INTERVAL == 0:
            pth = join(CHECKPOINT_PATH, str(ep))
            training_manager.save(pth)
            logger.info(f"All trainer states saved under {pth}")
        # performance details
        if ep == EVAL_SCHEDULE[eval_point_index]:
            logger.info(f"Eval {ep} starting")
            eval_point_index += 1
            result = env_sampler.eval()
            # if scenario.post_evaluate:
            #     scenario.post_evaluate(result["info"], ep)
            # tracker = result['tracker']
            # tracker.render(LOG_PATH, 'a_plot_balance.png', tracker.step_balances, ["OuterRetailerFacility"])
            # tracker.render(LOG_PATH, 'a_plot_reward.png', tracker.step_rewards, ["OuterRetailerFacility"])
            # tracker.render_sku(LOG_PATH)
            
            # df_product = pd.DataFrame(env_sampler._balance_calculator.product_metric_track)
            # df_product = df_product.groupby(['tick', 'id']).first().reset_index()
            # df_product.to_csv(f'{LOG_PATH}/output_product_metrics_{ep}.csv', index=False)
