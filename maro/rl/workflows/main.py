# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import importlib
import os
import sys
import time
from typing import List, Type

from maro.rl.rl_component.rl_component_bundle import RLComponentBundle
from maro.rl.rollout import BatchEnvSampler, ExpElement
from maro.rl.training import TrainingManager
from maro.rl.utils import get_torch_device
from maro.rl.utils.common import float_or_none, get_env, int_or_none, list_or_none
from maro.rl.utils.training import get_latest_ep
from maro.utils import LoggerV2


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MARO RL workflow parser")
    parser.add_argument("--evaluate_only", action="store_true", help="Only run evaluation part of the workflow")
    return parser.parse_args()


def main(rl_component_bundle: RLComponentBundle, args: argparse.Namespace) -> None:
    if args.evaluate_only:
        evaluate_only_workflow(rl_component_bundle)
    else:
        training_workflow(rl_component_bundle)


def training_workflow(rl_component_bundle: RLComponentBundle) -> None:
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

    is_single_thread = train_mode == "simple" and not parallel_rollout
    if is_single_thread:
        rl_component_bundle.pre_create_policy_instances()

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
        env_sampler = rl_component_bundle.env_sampler
        if train_mode != "simple":
            for policy_name, device_name in rl_component_bundle.device_mapping.items():
                env_sampler.assign_policy_to_device(policy_name, get_torch_device(device_name))

    # evaluation schedule
    eval_schedule = list_or_none(get_env("EVAL_SCHEDULE", required=False))
    logger.info(f"Policy will be evaluated at the end of episodes {eval_schedule}")
    eval_point_index = 0

    training_manager = TrainingManager(
        rl_component_bundle=rl_component_bundle,
        explicit_assign_device=(train_mode == "simple"),
        proxy_address=None
        if train_mode == "simple"
        else (
            get_env("TRAIN_PROXY_HOST"),
            int(get_env("TRAIN_PROXY_FRONTEND_PORT")),
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

        env_sampler.post_collect(total_info_list, ep)

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
                policy_state=training_manager.get_policy_state() if not is_single_thread else None,
            )
            env_sampler.post_evaluate(result["info"], ep)

    if isinstance(env_sampler, BatchEnvSampler):
        env_sampler.exit()
    training_manager.exit()


def evaluate_only_workflow(rl_component_bundle: RLComponentBundle) -> None:
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
        env_sampler = rl_component_bundle.env_sampler

    load_path = get_env("LOAD_PATH", required=False)
    load_episode = int_or_none(get_env("LOAD_EPISODE", required=False))
    if load_path:
        assert isinstance(load_path, str)

        ep = load_episode if load_episode is not None else get_latest_ep(load_path)
        path = os.path.join(load_path, str(ep))

        loaded = env_sampler.load_policy_state(path)
        logger.info(f"Loaded policies {loaded} into env sampler from {path}")

    result = env_sampler.eval()
    env_sampler.post_evaluate(result["info"], -1)

    if isinstance(env_sampler, BatchEnvSampler):
        env_sampler.exit()


if __name__ == "__main__":
    scenario_path = get_env("SCENARIO_PATH")
    scenario_path = os.path.normpath(scenario_path)
    sys.path.insert(0, os.path.dirname(scenario_path))
    module = importlib.import_module(os.path.basename(scenario_path))

    rl_component_bundle_cls: Type[RLComponentBundle] = getattr(module, "rl_component_bundle_cls")
    rl_component_bundle = rl_component_bundle_cls()
    main(rl_component_bundle, args=get_args())
