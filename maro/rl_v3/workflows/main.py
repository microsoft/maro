# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import time
from typing import List

from maro.rl_v3.rollout import BatchEnvSampler, ExpElement
from maro.rl_v3.training import SimpleTrainerManager
from maro.rl_v3.utils.common import from_env, from_env_as_int, get_eval_schedule, get_logger, get_module
from maro.rl_v3.workflows.utils import ScenarioAttr, _get_scenario_path

if __name__ == "__main__":
    # get user-defined scenario ingredients
    scenario = get_module(_get_scenario_path())
    scenario_attr = ScenarioAttr(scenario)
    agent2policy = scenario_attr.agent2policy
    policy_creator = scenario_attr.policy_creator
    trainer_creator = scenario_attr.trainer_creator
    post_collect = scenario_attr.post_collect
    post_evaluate = scenario_attr.post_evaluate

    rollout_mode, train_mode = str(from_env("ROLLOUT_MODE")), str(from_env("TRAIN_MODE"))
    assert rollout_mode in {"simple", "parallel"}
    assert train_mode in {"simple", "parallel"}
    if train_mode == "parallel":
        dispatcher_address = (from_env("DISPATCHER_HOST"), from_env("DISPATCHER_FRONTEND_PORT"))
    else:
        dispatcher_address = None

    if train_mode == "simple" and rollout_mode == "simple":
        policy_dict = {name: get_policy_func(name) for name, get_policy_func in policy_creator.items()}
        policy_creator = {name: lambda name: policy_dict[name] for name in policy_dict}

    env_sampler = scenario_attr.get_env_sampler()

    num_episodes = from_env_as_int("NUM_EPISODES")
    num_steps = from_env_as_int("NUM_STEPS", required=False, default=-1)

    load_path = from_env("LOAD_PATH", required=False, default=None)
    checkpoint_path = from_env("CHECKPOINT_PATH", required=False, default=None)
    log_path = str(from_env("LOG_PATH", required=False, default=os.getcwd()))
    job_path = str(from_env("JOB"))
    logger = get_logger(log_path, job_path, "MAIN")

    # evaluation schedule
    eval_schedule_config = from_env("EVAL_SCHEDULE", required=False, default=None)
    assert isinstance(eval_schedule_config, int) or isinstance(eval_schedule_config, list)
    eval_schedule = get_eval_schedule(eval_schedule_config, num_episodes)
    logger.info(f"Policy will be evaluated at the end of episodes {eval_schedule}")
    eval_point_index = 0

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
            if train_mode == "parallel":
                policy_states = {
                    policy_name: state
                    for policy_state in trainer_manager.get_policy_states().values()
                    for policy_name, state in policy_state.items()
                }
                env_sampler.set_policy_states(policy_states)

            result = env_sampler.sample(num_steps=num_steps)
            experiences: List[ExpElement] = result["experiences"]
            trackers = [result["tracker"]]
            logger.info(f"Roll-out finished (episode: {ep})")
            end_of_episode: bool = result["end_of_episode"]

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
                policy_name: state
                for policy_state in trainer_manager.get_policy_states().values()
                for policy_name, state in policy_state.items()
            }
            trackers = env_sampler.test(policy_states)
            if post_evaluate:
                post_evaluate([trackers], ep)
