# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import time

from maro.rl.learning.helpers import get_rollout_finish_msg
from maro.rl.workflows.helpers import from_env, get_eval_schedule, get_log_dir, get_scenario_module
from maro.utils import Logger


if __name__ == "__main__":
    # get user-defined scenario ingredients
    scenario = get_scenario_module(from_env("SCENARIODIR"))
    get_env_sampler = getattr(scenario, "get_env_sampler")
    post_collect = getattr(scenario, "post_collect", None)
    post_evaluate = getattr(scenario, "post_evaluate", None)

    mode = from_env("MODE")
    num_episodes = from_env("NUMEPISODES")
    num_steps = from_env("NUMSTEPS", required=False, default=-1)

    load_policy_dir = from_env("LOADDIR", required=False, default=None)
    checkpoint_dir = from_env("CHECKPOINTDIR", required=False, default=None)
    log_dir = get_log_dir(from_env("LOGDIR", required=False, default=os.getcwd()), from_env("JOB"))
    logger = Logger("MAIN", dump_folder=log_dir)

    # evaluation schedule
    eval_schedule = get_eval_schedule(from_env("EVALSCH", required=False, default=None), num_episodes)
    logger.info(f"Policy will be evaluated at the end of episodes {eval_schedule}")
    eval_point_index = 0
    if mode == "single":
        env_sampler = get_env_sampler()
        if load_policy_dir:
            env_sampler.agent_wrapper.load(load_policy_dir)
            logger.info(f"Loaded policy states from {load_policy_dir}")

        for ep in range(1, num_episodes + 1):
            collect_time = policy_update_time = 0
            segment, end_of_episode = 1, False
            while not end_of_episode:
                # experience collection
                tc0 = time.time()
                result = env_sampler.sample(num_steps=num_steps, return_rollout_info=False)
                trackers = [result["tracker"]]
                logger.info(
                    get_rollout_finish_msg(ep, result["step_range"], exploration_params=result["exploration_params"])
                )
                end_of_episode = result["end_of_episode"]

                if post_collect:
                    post_collect(trackers, ep, segment)

                collect_time += time.time() - tc0
                tu0 = time.time()
                env_sampler.agent_wrapper.improve(checkpoint_dir=checkpoint_dir)
                if checkpoint_dir:
                    logger.info(f"Saved policy states to {checkpoint_dir}")
                policy_update_time += time.time() - tu0
                segment += 1

            # performance details
            logger.info(f"ep {ep} summary - collect time: {collect_time}, policy update time: {policy_update_time}")
            if eval_schedule and ep == eval_schedule[eval_point_index]:
                eval_point_index += 1
                trackers = [env_sampler.test()]
                if post_evaluate:
                    post_evaluate(trackers, ep)
    else:
        from policy_manager import get_policy_manager
        from rollout_manager import get_rollout_manager

        rollout_manager = get_rollout_manager()
        policy_manager = get_policy_manager()
        for ep in range(1, num_episodes + 1):
            collect_time = policy_update_time = 0
            rollout_manager.reset()
            segment, end_of_episode = 1, False
            while not end_of_episode:
                # experience collection
                tc0 = time.time()
                policy_state_dict = policy_manager.get_state()
                rollout_info_by_policy, trackers = rollout_manager.collect(ep, segment, policy_state_dict)
                end_of_episode = rollout_manager.end_of_episode

                if post_collect:
                    post_collect(trackers, ep, segment)

                collect_time += time.time() - tc0
                tu0 = time.time()
                policy_manager.update(rollout_info_by_policy)
                policy_update_time += time.time() - tu0
                segment += 1

            # performance details
            logger.info(f"ep {ep} summary - collect time: {collect_time}, policy update time: {policy_update_time}")
            if eval_schedule and ep == eval_schedule[eval_point_index]:
                eval_point_index += 1
                trackers = rollout_manager.evaluate(ep, policy_manager.get_state())
                if post_evaluate:
                    post_evaluate(trackers, ep)

        rollout_manager.exit()
        if hasattr(policy_manager, "exit"):
            policy_manager.exit()
