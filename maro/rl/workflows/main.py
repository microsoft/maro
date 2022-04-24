# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import os
import time
from typing import List

from maro.rl.rollout import BatchEnvSampler, ExpElement
from maro.rl.training import TrainingManager
from maro.rl.training.utils import get_latest_ep
from maro.rl.utils import get_torch_device
from maro.rl.utils.common import float_or_none, get_env, int_or_none, list_or_none
from maro.rl.workflows.scenario import Scenario
from maro.utils import LoggerV2


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MARO RL workflow parser")
    parser.add_argument("--evaluate_only", action="store_true", help="Only run evaluation part of the workflow")
    return parser.parse_args()


def main(scenario: Scenario, args: argparse.Namespace) -> None:
    if args.evaluate_only:
        evaluate_only_workflow(scenario)
    else:
        training_workflow(scenario)


def training_workflow(scenario: Scenario) -> None:
    num_episodes = int(get_env("NUM_EPISODES"))
    num_steps = int_or_none(get_env("NUM_STEPS", required=False))
    min_n_sample = int_or_none(get_env("MIN_N_SAMPLE"))

    logger = LoggerV2(
        "MAIN",
        dump_path=get_env("LOG_PATH"),
        dump_mode="a",
        stdout_level=get_env("LOG_LEVEL_STDOUT", required=False, default="CRITICAL"),
        file_level=get_env("LOG_LEVEL_FILE", required=False, default="CRITICAL"),
    )
    logger.info("Start training workflow.")

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
                env_sampler.rl_policy_dict[policy_name].to_device(get_torch_device(device_name))

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
    load_episode = int_or_none(get_env("LOAD_EPISODE", required=False))
    if load_path:
        assert isinstance(load_path, str)

        ep = load_episode if load_episode is not None else get_latest_ep(load_path)
        path = os.path.join(load_path, str(ep))

        loaded = env_sampler.load_policy_state(path)
        logger.info(f"Loaded policies {loaded} into env sampler from {path}")

        loaded = training_manager.load(path)
        logger.info(f"Loaded trainers {loaded} from {path}")
        start_ep = ep + 1
    else:
        start_ep = 1

    checkpoint_path = get_env("CHECKPOINT_PATH", required=False)
    checkpoint_interval = int_or_none(get_env("CHECKPOINT_INTERVAL", required=False))

    # main loop
    for ep in range(start_ep, num_episodes + 1):
        collect_time = training_time = 0
        total_experiences: List[List[ExpElement]] = []
        total_info_list: List[dict] = []
        n_sample = 0
        while n_sample < min_n_sample:
            tc0 = time.time()
            result = env_sampler.sample(
                policy_state=training_manager.get_policy_state() if not is_single_thread else None,
                num_steps=num_steps,
            )
            experiences: List[List[ExpElement]] = result["experiences"]
            info_list: List[dict] = result["info"]

            n_sample += len(experiences[0])
            total_experiences.extend(experiences)
            total_info_list.extend(info_list)

            collect_time += time.time() - tc0

        if scenario.post_collect:
            scenario.post_collect(total_info_list, ep, -1)  # TODO

        logger.info(f"Roll-out completed for episode {ep}. Training started...")
        tu0 = time.time()
        training_manager.record_experiences(total_experiences)
        training_manager.train_step()
        if checkpoint_path and (checkpoint_interval is None or ep % checkpoint_interval == 0):
            assert isinstance(checkpoint_path, str)
            pth = os.path.join(checkpoint_path, str(ep))
            training_manager.save(pth)
            logger.info(f"All trainer states saved under {pth}")
        training_time += time.time() - tu0

        # performance details
        logger.info(f"ep {ep} - roll-out time: {collect_time:.2f} seconds, training time: {training_time:.2f} seconds")
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


def evaluate_only_workflow(scenario: Scenario) -> None:
    logger = LoggerV2(
        "MAIN",
        dump_path=get_env("LOG_PATH"),
        dump_mode="a",
        stdout_level=get_env("LOG_LEVEL_STDOUT", required=False, default="CRITICAL"),
        file_level=get_env("LOG_LEVEL_FILE", required=False, default="CRITICAL"),
    )
    logger.info("Start evaluate only workflow.")

    env_sampling_parallelism = int_or_none(get_env("ENV_SAMPLE_PARALLELISM", required=False))
    env_eval_parallelism = int_or_none(get_env("ENV_EVAL_PARALLELISM", required=False))
    parallel_rollout = env_sampling_parallelism is not None or env_eval_parallelism is not None

    policy_creator = scenario.policy_creator
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

    load_path = get_env("LOAD_PATH", required=False)
    load_episode = int_or_none(get_env("LOAD_EPISODE", required=False))
    if load_path:
        assert isinstance(load_path, str)

        ep = load_episode if load_episode is not None else get_latest_ep(load_path)
        path = os.path.join(load_path, str(ep))

        loaded = env_sampler.load_policy_state(path)
        logger.info(f"Loaded policies {loaded} into env sampler from {path}")

    result = env_sampler.eval()
    if scenario.post_evaluate:
        scenario.post_evaluate(result["info"], -1)

    if isinstance(env_sampler, BatchEnvSampler):
        env_sampler.exit()


if __name__ == "__main__":
    # get user-defined scenario ingredients
    run_scenario = Scenario(get_env("SCENARIO_PATH"))
    main(run_scenario, args=get_args())
