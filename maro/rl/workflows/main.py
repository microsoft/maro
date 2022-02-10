# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import time
from typing import List

from maro.rl.rollout import BatchEnvSampler, ExpElement
from maro.rl.training import TrainerManager
from maro.rl.utils.common import from_env, from_env_as_int, get_eval_schedule
from maro.rl.workflows.scenario import Scenario
from maro.utils import Logger


def main(scenario: Scenario):
    num_episodes = from_env_as_int("NUM_EPISODES")
    num_steps = from_env_as_int("NUM_STEPS", required=False, default=-1)

    logger = Logger(
        "MAIN",
        dump_path=from_env("LOG_PATH"),
        dump_mode="a",
        stdout_level=from_env("LOG_LEVEL_STDOUT", required=False, default="CRITICAL"),
        file_level=from_env("LOG_LEVEL_FILE", required=False, default="CRITICAL")
    )

    load_path = from_env("LOAD_PATH", required=False, default=None)
    checkpoint_path = from_env("CHECKPOINT_PATH", required=False, default=None)
    checkpoint_interval = from_env_as_int("CHECKPOINT_INTERVAL", required=False, default=1)

    env_sampling_parallelism = from_env_as_int("ENV_SAMPLE_PARALLELISM", required=False, default=1)
    env_eval_parallelism = from_env_as_int("ENV_EVAL_PARALLELISM", required=False, default=1)
    rollout_parallelism = max(env_sampling_parallelism, env_eval_parallelism)
    train_mode = from_env("TRAIN_MODE")

    agent2policy = scenario.agent2policy
    policy_creator = scenario.policy_creator
    trainer_creator = scenario.trainer_creator
    is_single_thread = train_mode == "simple" and rollout_parallelism == 1
    if is_single_thread:
        # If running in single thread mode, create policy instances here and reuse then in rollout and training.
        policy_dict = {name: get_policy_func(name) for name, get_policy_func in policy_creator.items()}
        policy_creator = {name: lambda name: policy_dict[name] for name in policy_dict}

    if rollout_parallelism == 1:
        env_sampler = scenario.get_env_sampler(policy_creator)
    else:
        env_sampler = BatchEnvSampler(
            sampling_parallelism=env_sampling_parallelism,
            port=from_env_as_int("ROLLOUT_CONTROLLER_PORT"),
            min_env_samples=from_env("MIN_ENV_SAMPLES", required=False, default=None),
            grace_factor=from_env("GRACE_FACTOR", required=False, default=None),
            eval_parallelism=env_eval_parallelism,
            logger=logger
        )

    # evaluation schedule
    eval_schedule_config = from_env("EVAL_SCHEDULE", required=False, default=None)
    assert isinstance(eval_schedule_config, int) or isinstance(eval_schedule_config, list)
    eval_schedule = get_eval_schedule(eval_schedule_config, num_episodes)
    logger.info(f"Policy will be evaluated at the end of episodes {eval_schedule}")
    eval_point_index = 0

    trainer_manager = TrainerManager(
        policy_creator, trainer_creator, agent2policy,
        proxy_address=None if train_mode == "simple" else (
            from_env("TRAIN_PROXY_HOST"), from_env_as_int("TRAIN_PROXY_FRONTEND_PORT")
        ),
        logger=logger
    )
    if load_path:
        loaded = trainer_manager.load(load_path)
        logger.info(f"Loaded states for {loaded} from {load_path}")

    # main loop
    for ep in range(1, num_episodes + 1):
        collect_time = training_time = 0
        segment, end_of_episode = 1, False
        while not end_of_episode:
            # Experience collection
            tc0 = time.time()
            result = env_sampler.sample(
                policy_state=trainer_manager.get_policy_state() if not is_single_thread else None,
                num_steps=num_steps
            )
            experiences: List[List[ExpElement]] = result["experiences"]
            end_of_episode: bool = result["end_of_episode"]

            if scenario.post_collect:
                scenario.post_collect(result["info"], ep, segment)

            collect_time += time.time() - tc0

            logger.info(f"Roll-out completed for episode {ep}. Training started...")
            tu0 = time.time()
            trainer_manager.record_experiences(experiences)
            trainer_manager.train()
            if checkpoint_path and ep % checkpoint_interval == 0:
                pth = os.path.join(checkpoint_path, str(ep))
                trainer_manager.save(pth)
                logger.info(f"All trainer states saved under {pth}")
            training_time += time.time() - tu0
            segment += 1

        # performance details
        logger.info(f"ep {ep} - roll-out time: {collect_time}, training time: {training_time}")
        if eval_schedule and ep == eval_schedule[eval_point_index]:
            eval_point_index += 1
            result = env_sampler.eval(
                policy_state=trainer_manager.get_policy_state() if not is_single_thread else None
            )
            if scenario.post_evaluate:
                scenario.post_evaluate(result["info"], ep)

    if isinstance(env_sampler, BatchEnvSampler):
        env_sampler.exit()
    trainer_manager.exit()


if __name__ == "__main__":
    # get user-defined scenario ingredients
    scenario = Scenario(from_env("SCENARIO_PATH"))
    main(scenario)
