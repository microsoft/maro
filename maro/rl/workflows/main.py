# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import time
from typing import List

from maro.rl.rollout import BatchEnvSampler, ExpElement
from maro.rl.training import TrainerManager
from maro.rl.utils.common import from_env, from_env_as_float, from_env_as_int, get_eval_schedule, get_logger, get_module
from maro.rl.workflows.utils import ScenarioAttr, _get_scenario_path

if __name__ == "__main__":
    # get user-defined scenario ingredients
    scenario = get_module(_get_scenario_path())
    scenario_attr = ScenarioAttr(scenario)
    agent2policy = scenario_attr.agent2policy
    policy_creator = scenario_attr.policy_creator
    trainer_creator = scenario_attr.trainer_creator
    post_collect = scenario_attr.post_collect
    post_evaluate = scenario_attr.post_evaluate

    num_episodes = from_env_as_int("NUM_EPISODES")
    num_steps = from_env_as_int("NUM_STEPS", required=False, default=-1)

    load_path = from_env("LOAD_PATH", required=False, default=None)
    checkpoint_path = from_env("CHECKPOINT_PATH", required=False, default=None)
    log_path = str(from_env("LOG_PATH", required=False, default=os.getcwd()))
    logger = get_logger(log_path, str(from_env("JOB")), "MAIN")

    rollout_mode, train_mode = str(from_env("ROLLOUT_MODE")), str(from_env("TRAIN_MODE"))
    assert rollout_mode in {"simple", "parallel"} and train_mode in {"simple", "parallel"}
    if train_mode == "parallel":
        dispatcher_address = (from_env("DISPATCHER_HOST"), from_env_as_int("DISPATCHER_FRONTEND_PORT"))
    else:
        dispatcher_address = None

    is_single_thread = train_mode == "simple" and rollout_mode == "simple"
    if is_single_thread:
        policy_dict = {name: get_policy_func(name) for name, get_policy_func in policy_creator.items()}
        policy_creator = {name: lambda name: policy_dict[name] for name in policy_dict}

    if rollout_mode == "simple":
        env_sampler = scenario_attr.get_env_sampler(policy_creator)
    else:
        env_sampler = BatchEnvSampler(
            parallelism=from_env_as_int("ROLLOUT_PARALLELISM"),
            remote_address=(str(from_env("ROLLOUT_PROXY_HOST")), from_env_as_int("ROLLOUT_PROXY_FRONTEND_PORT")),
            min_env_samples=from_env_as_int("MIN_ENV_SAMPLES", required=False, default=None),
            grace_factor=from_env_as_float("GRACE_FACTOR", required=False, default=None),
            eval_parallelism=from_env_as_int("EVAL_PARALLELISM", required=False, default=1),
            logger=logger
        )

    # evaluation schedule
    eval_schedule_config = from_env("EVAL_SCHEDULE", required=False, default=None)
    assert isinstance(eval_schedule_config, int) or isinstance(eval_schedule_config, list)
    eval_schedule = get_eval_schedule(eval_schedule_config, num_episodes)
    logger.info(f"Policy will be evaluated at the end of episodes {eval_schedule}")
    eval_point_index = 0

    trainer_manager = TrainerManager(
        policy_creator, trainer_creator, agent2policy, dispatcher_address=dispatcher_address, logger=logger
    )
    if load_path:
        loaded = trainer_manager.load(load_path)
        logger.info(f"Loaded states for {loaded} from {load_path}")

    # main loop
    for ep in range(1, num_episodes + 1):
        collect_time = training_time = 0
        segment, end_of_episode = 1, False
        while not end_of_episode:
            # experience collection
            tc0 = time.time()
            policy_state = trainer_manager.get_policy_state() if not is_single_thread else None
            result = env_sampler.sample(policy_state=policy_state, num_steps=num_steps)
            experiences: List[List[ExpElement]] = result["experiences"]
            end_of_episode: bool = result["end_of_episode"]

            if post_collect:
                post_collect(result["info"], ep, segment)

            collect_time += time.time() - tc0

            logger.info(f"Roll-out completed for episode {ep}. Training started...")
            tu0 = time.time()
            trainer_manager.record_experiences(experiences)
            trainer_manager.train()
            if checkpoint_path:
                pth = os.path.join(checkpoint_path, str(ep))
                trainer_manager.save(pth)
                logger.info(f"All trainer states saved under {pth}")
            training_time += time.time() - tu0
            segment += 1

        # performance details
        logger.info(f"ep {ep} - roll-out time: {collect_time}, training time: {training_time}")
        if eval_schedule and ep == eval_schedule[eval_point_index]:
            eval_point_index += 1
            policy_state = trainer_manager.get_policy_state() if not is_single_thread else None
            result = env_sampler.test(policy_state=policy_state)
            if post_evaluate:
                post_evaluate(result["info"], ep)
