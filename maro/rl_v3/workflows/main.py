# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import time

from maro.rl_v3.rollout.helpers import get_rollout_finish_msg
from maro.rl_v3.training.trainer_manager import SimpleTrainerManager
from maro.rl_v3.utils.common import from_env, get_eval_schedule, get_logger, get_module


if __name__ == "__main__":
    # get user-defined scenario ingredients
    scenario = get_module(from_env("SCENARIO_PATH"))
    get_env_sampler = getattr(scenario, "get_env_sampler")
    agent2policy = getattr(scenario, "agent2policy")
    policy_creator = getattr(scenario, "policy_creator")
    trainer_creator = getattr(scenario, "trainer_creator")
    post_collect = getattr(scenario, "post_collect", None)
    post_evaluate = getattr(scenario, "post_evaluate", None)

    mode = from_env("MODE")
    env_sampler = get_env_sampler()
    num_episodes = from_env("NUM_EPISODES")
    num_steps = from_env("NUM_STEPS", required=False, default=-1)

    load_path = from_env("LOAD_PATH", required=False, default=None)
    checkpoint_path = from_env("CHECKPOINT_PATH", required=False, default=None)
    logger = get_logger(from_env("LOG_PATH", required=False, default=os.getcwd()), from_env("JOB"), "MAIN")

    # evaluation schedule
    eval_schedule = get_eval_schedule(from_env("EVAL_SCHEDULE", required=False, default=None), num_episodes)
    logger.info(f"Policy will be evaluated at the end of episodes {eval_schedule}")
    eval_point_index = 0

    train_mode = from_env("TRAINING_MODE", required=True)
    if train_mode not in {"simple", "parallel"}:
        raise ValueError(f"Unsupported training mode: {train_mode}. Supported modes: simple, parallel")

    # Batching trainers for parallelism
    if train_mode == "parallel":
        dispatcher_address = (from_env("DISPATCHER_HOST"), from_env("DISPATCHER_FRONTEND_PORT"))
    else:
        dispatcher_address = None
    trainer_manager = SimpleTrainerManager(
        policy_creator, trainer_creator, agent2policy, dispatcher_address=dispatcher_address
    )

    # main loop
    for ep in range(1, num_episodes + 1):
        collect_time = training_time = 0
        segment, end_of_episode = 1, False
        while not end_of_episode:
            # experience collection
            tc0 = time.time()
            policy_states = {
                policy_name: state for policy_state in trainer_manager.get_policy_states().values()
                for policy_name, state in policy_state.items()
            }
            env_sampler.set_policy_states(policy_states)
            result = env_sampler.sample(num_steps=num_steps)
            experiences = result["experiences"]
            trackers = [result["tracker"]]
            logger.info(get_rollout_finish_msg(ep))
            end_of_episode = result["end_of_episode"]

            if post_collect:
                post_collect(trackers, ep, segment)
            
            collect_time += time.time() - tc0

            tu0 = time.time()
            trainer_manager.record_experiences(experiences)
            trainer_manager.train()
            training_time += time.time() - tu0
            segment += 1

        # performance details
        logger.info(f"ep {ep} summary - collect time: {collect_time}, policy update time: {training_time}")
        if eval_schedule and ep == eval_schedule[eval_point_index]:
            eval_point_index += 1
            policy_states = {
                policy_name: state for policy_state in trainer_manager.get_policy_states().values()
                for policy_name, state in policy_state.items()
            }
            trackers = env_sampler.test(policy_states)
            if post_evaluate:
                post_evaluate([trackers], ep)
    
    for trainer_name, trainer in trainer_manager._trainer_dict.items():
        logger.info(f"trainer_name = {trainer_name}, retries = {trainer._ops._client._retries}")
