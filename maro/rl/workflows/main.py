# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import time
from typing import List

from maro.rl.rollout import BatchEnvSampler, ExpElement
from maro.rl.training import TrainingManager
from maro.rl.utils.common import float_or_none, get_env, int_or_none, list_or_none
from maro.rl.workflows.scenario import Scenario
from maro.utils import LoggerV2


def main(scenario: Scenario) -> None:
    num_episodes = int(get_env("NUM_EPISODES"))
    num_steps = int_or_none(get_env("NUM_STEPS", required=False))

    logger = LoggerV2(
        "MAIN",
        dump_path=get_env("LOG_PATH"),
        dump_mode="a",
        stdout_level=get_env("LOG_LEVEL_STDOUT", required=False, default="CRITICAL"),
        file_level=get_env("LOG_LEVEL_FILE", required=False, default="CRITICAL"),
    )

    env_sampling_parallelism = int_or_none(get_env("ENV_SAMPLE_PARALLELISM", required=False))
    env_eval_parallelism = int_or_none(get_env("ENV_EVAL_PARALLELISM", required=False))
    parallel_rollout = env_sampling_parallelism is not None or env_eval_parallelism is not None
    train_mode = get_env("TRAIN_MODE")

    agent2policy = scenario.agent2policy
    policy_creator = scenario.policy_creator
    trainer_creator = scenario.trainer_creator
    is_single_thread = train_mode == "simple" and not parallel_rollout
    if is_single_thread:
        # If running in single thread mode, create policy instances here and reuse then in rollout and training.
        # In other words, `policy_creator` will return a policy instance that has been already created in advance
        # instead of create a new policy instance.
        policy_dict = {name: get_policy_func(name) for name, get_policy_func in policy_creator.items()}
        policy_creator = {name: lambda name: policy_dict[name] for name in policy_dict}

    if parallel_rollout:
        env_sampler = BatchEnvSampler(
            sampling_parallelism=env_sampling_parallelism,
            port=int(get_env("ROLLOUT_CONTROLLER_PORT")),
            min_env_samples=int_or_none(get_env("MIN_ENV_SAMPLES", required=False)),
            grace_factor=float_or_none(get_env("GRACE_FACTOR", required=False)),
            eval_parallelism=env_eval_parallelism,
            logger=logger,
        )
    else:
        env_sampler = scenario.env_sampler_creator(policy_creator)
        if train_mode != "simple":
            for policy_name, device_name in scenario.device_mapping.items():
                env_sampler.rl_policy_dict[policy_name].to_device(device_name)

    # evaluation schedule
    eval_schedule = list_or_none(get_env("EVAL_SCHEDULE", required=False))
    logger.info(f"Policy will be evaluated at the end of episodes {eval_schedule}")
    eval_point_index = 0

    if scenario.trainable_policies is None:
        trainable_policies = set(policy_creator.keys())
    else:
        trainable_policies = set(scenario.trainable_policies)

    trainable_policy_creator = {name: func for name, func in policy_creator.items() if name in trainable_policies}
    trainable_agent2policy = {id_: name for id_, name in agent2policy.items() if name in trainable_policies}
    training_manager = TrainingManager(
        policy_creator=trainable_policy_creator,
        trainer_creator=trainer_creator,
        agent2policy=trainable_agent2policy,
        device_mapping=scenario.device_mapping if train_mode == "simple" else {},
        proxy_address=None if train_mode == "simple" else (
            get_env("TRAIN_PROXY_HOST"), int(get_env("TRAIN_PROXY_FRONTEND_PORT"))
        ),
        logger=logger,
    )

    load_path = get_env("LOAD_PATH", required=False)
    if load_path:
        assert isinstance(load_path, str)
        loaded = training_manager.load(load_path)
        logger.info(f"Loaded states for {loaded} from {load_path}")

    checkpoint_path = get_env("CHECKPOINT_PATH", required=False)
    checkpoint_interval = int_or_none(get_env("CHECKPOINT_INTERVAL", required=False))
    # main loop
    for ep in range(1, num_episodes + 1):
        collect_time = training_time = 0
        segment, end_of_episode = 1, False
        while not end_of_episode:
            # Experience collection
            tc0 = time.time()
            result = env_sampler.sample(
                policy_state=training_manager.get_policy_state() if not is_single_thread else None,
                num_steps=num_steps,
            )
            experiences: List[List[ExpElement]] = result["experiences"]
            end_of_episode: bool = result["end_of_episode"]

            if scenario.post_collect:
                scenario.post_collect(result["info"], ep, segment)

            collect_time += time.time() - tc0

            logger.info(f"Roll-out completed for episode {ep}, segment {segment}. Training started...")
            tu0 = time.time()
            training_manager.record_experiences(experiences)
            training_manager.train_step()
            if checkpoint_path and (checkpoint_interval is None or ep % checkpoint_interval == 0):
                assert isinstance(checkpoint_path, str)
                pth = os.path.join(checkpoint_path, str(ep))
                training_manager.save(pth)
                logger.info(f"All trainer states saved under {pth}")
            training_time += time.time() - tu0
            segment += 1

        # performance details
        logger.info(f"ep {ep} - roll-out time: {collect_time}, training time: {training_time}")
        if eval_schedule and ep == eval_schedule[eval_point_index]:
            eval_point_index += 1
            result = env_sampler.eval(
                policy_state=training_manager.get_policy_state() if not is_single_thread else None
            )
            if scenario.post_evaluate:
                scenario.post_evaluate(result["info"], ep)

    if isinstance(env_sampler, BatchEnvSampler):
        env_sampler.exit()
    training_manager.exit()


if __name__ == "__main__":
    # get user-defined scenario ingredients
    scenario = Scenario(get_env("SCENARIO_PATH"))
    main(scenario)
