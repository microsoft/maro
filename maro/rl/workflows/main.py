# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import importlib
import os
import sys
import time
from typing import List, Union

from maro.rl.rl_component.rl_component_bundle import RLComponentBundle
from maro.rl.rollout import AbsEnvSampler, BatchEnvSampler, ExpElement
from maro.rl.training import TrainingManager
from maro.rl.utils import get_torch_device
from maro.rl.utils.common import float_or_none, get_env, int_or_none, list_or_none
from maro.rl.utils.training import get_latest_ep
from maro.rl.workflows.utils import env_str_helper
from maro.utils import LoggerV2


class WorkflowEnvAttributes:
    def __init__(self) -> None:
        # Number of training episodes
        self.num_episodes = int(env_str_helper(get_env("NUM_EPISODES")))

        # Maximum number of steps in on round of sampling.
        self.num_steps = int_or_none(get_env("NUM_STEPS", required=False))

        # Minimum number of data samples to start a round of training. If the data samples are insufficient, re-run
        # data sampling until we have at least `min_n_sample` data entries.
        self.min_n_sample = int(env_str_helper(get_env("MIN_N_SAMPLE")))

        # Path to store logs.
        self.log_path = get_env("LOG_PATH")

        # Log levels
        self.log_level_stdout = get_env("LOG_LEVEL_STDOUT", required=False, default="CRITICAL")
        self.log_level_file = get_env("LOG_LEVEL_FILE", required=False, default="CRITICAL")

        # Parallelism of sampling / evaluation. Used in distributed sampling.
        self.env_sampling_parallelism = int_or_none(get_env("ENV_SAMPLE_PARALLELISM", required=False))
        self.env_eval_parallelism = int_or_none(get_env("ENV_EVAL_PARALLELISM", required=False))

        # Training mode, simple or distributed
        self.train_mode = get_env("TRAIN_MODE")

        # Evaluating schedule.
        self.eval_schedule = list_or_none(get_env("EVAL_SCHEDULE", required=False))

        # Restore configurations.
        self.load_path = get_env("LOAD_PATH", required=False)
        self.load_episode = int_or_none(get_env("LOAD_EPISODE", required=False))

        # Checkpointing configurations.
        self.checkpoint_path = get_env("CHECKPOINT_PATH", required=False)
        self.checkpoint_interval = int_or_none(get_env("CHECKPOINT_INTERVAL", required=False))

        # Parallel sampling configurations.
        self.parallel_rollout = self.env_sampling_parallelism is not None or self.env_eval_parallelism is not None
        if self.parallel_rollout:
            self.port = int(env_str_helper(get_env("ROLLOUT_CONTROLLER_PORT")))
            self.min_env_samples = int_or_none(get_env("MIN_ENV_SAMPLES", required=False))
            self.grace_factor = float_or_none(get_env("GRACE_FACTOR", required=False))

        self.is_single_thread = self.train_mode == "simple" and not self.parallel_rollout

        # Distributed training configurations.
        if self.train_mode != "simple":
            self.proxy_address = (
                env_str_helper(get_env("TRAIN_PROXY_HOST")),
                int(env_str_helper(get_env("TRAIN_PROXY_FRONTEND_PORT"))),
            )

        self.logger = LoggerV2(
            "MAIN",
            dump_path=self.log_path,
            dump_mode="a",
            stdout_level=self.log_level_stdout,
            file_level=self.log_level_file,
        )


def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MARO RL workflow parser")
    parser.add_argument("--evaluate_only", action="store_true", help="Only run evaluation part of the workflow")
    return parser.parse_args()


def _get_env_sampler(
    rl_component_bundle: RLComponentBundle,
    env_attr: WorkflowEnvAttributes,
) -> Union[AbsEnvSampler, BatchEnvSampler]:
    if env_attr.parallel_rollout:
        assert env_attr.env_sampling_parallelism is not None
        return BatchEnvSampler(
            sampling_parallelism=env_attr.env_sampling_parallelism,
            port=env_attr.port,
            min_env_samples=env_attr.min_env_samples,
            grace_factor=env_attr.grace_factor,
            eval_parallelism=env_attr.env_eval_parallelism,
            logger=env_attr.logger,
        )
    else:
        env_sampler = rl_component_bundle.env_sampler
        if rl_component_bundle.device_mapping is not None:
            for policy_name, device_name in rl_component_bundle.device_mapping.items():
                env_sampler.assign_policy_to_device(policy_name, get_torch_device(device_name))
        return env_sampler


def main(rl_component_bundle: RLComponentBundle, env_attr: WorkflowEnvAttributes, args: argparse.Namespace) -> None:
    if args.evaluate_only:
        evaluate_only_workflow(rl_component_bundle, env_attr)
    else:
        training_workflow(rl_component_bundle, env_attr)


def training_workflow(rl_component_bundle: RLComponentBundle, env_attr: WorkflowEnvAttributes) -> None:
    env_attr.logger.info("Start training workflow.")

    env_sampler = _get_env_sampler(rl_component_bundle, env_attr)

    # evaluation schedule
    env_attr.logger.info(f"Policy will be evaluated at the end of episodes {env_attr.eval_schedule}")
    eval_point_index = 0

    training_manager = TrainingManager(
        rl_component_bundle=rl_component_bundle,
        explicit_assign_device=(env_attr.train_mode == "simple"),
        proxy_address=None if env_attr.train_mode == "simple" else env_attr.proxy_address,
        logger=env_attr.logger,
    )

    if env_attr.load_path:
        assert isinstance(env_attr.load_path, str)

        ep = env_attr.load_episode if env_attr.load_episode is not None else get_latest_ep(env_attr.load_path)
        path = os.path.join(env_attr.load_path, str(ep))

        loaded = env_sampler.load_policy_state(path)
        env_attr.logger.info(f"Loaded policies {loaded} into env sampler from {path}")

        loaded = training_manager.load(path)
        env_attr.logger.info(f"Loaded trainers {loaded} from {path}")
        start_ep = ep + 1
    else:
        start_ep = 1

    # main loop
    for ep in range(start_ep, env_attr.num_episodes + 1):
        collect_time = training_time = 0.0
        total_experiences: List[List[ExpElement]] = []
        total_info_list: List[dict] = []
        n_sample = 0
        while n_sample < env_attr.min_n_sample:
            tc0 = time.time()
            result = env_sampler.sample(
                policy_state=training_manager.get_policy_state() if not env_attr.is_single_thread else None,
                num_steps=env_attr.num_steps,
            )
            experiences: List[List[ExpElement]] = result["experiences"]
            info_list: List[dict] = result["info"]

            n_sample += len(experiences[0])
            total_experiences.extend(experiences)
            total_info_list.extend(info_list)

            collect_time += time.time() - tc0

        env_sampler.post_collect(total_info_list, ep)

        env_attr.logger.info(f"Roll-out completed for episode {ep}. Training started...")
        tu0 = time.time()
        training_manager.record_experiences(total_experiences)
        training_manager.train_step()
        if env_attr.checkpoint_path and (not env_attr.checkpoint_interval or ep % env_attr.checkpoint_interval == 0):
            assert isinstance(env_attr.checkpoint_path, str)
            pth = os.path.join(env_attr.checkpoint_path, str(ep))
            training_manager.save(pth)
            env_attr.logger.info(f"All trainer states saved under {pth}")
        training_time += time.time() - tu0

        # performance details
        env_attr.logger.info(
            f"ep {ep} - roll-out time: {collect_time:.2f} seconds, training time: {training_time:.2f} seconds",
        )
        if env_attr.eval_schedule and ep == env_attr.eval_schedule[eval_point_index]:
            eval_point_index += 1
            result = env_sampler.eval(
                policy_state=training_manager.get_policy_state() if not env_attr.is_single_thread else None,
            )
            env_sampler.post_evaluate(result["info"], ep)

    if isinstance(env_sampler, BatchEnvSampler):
        env_sampler.exit()
    training_manager.exit()


def evaluate_only_workflow(rl_component_bundle: RLComponentBundle, env_attr: WorkflowEnvAttributes) -> None:
    env_attr.logger.info("Start evaluate only workflow.")

    env_sampler = _get_env_sampler(rl_component_bundle, env_attr)

    if env_attr.load_path:
        assert isinstance(env_attr.load_path, str)

        ep = env_attr.load_episode if env_attr.load_episode is not None else get_latest_ep(env_attr.load_path)
        path = os.path.join(env_attr.load_path, str(ep))

        loaded = env_sampler.load_policy_state(path)
        env_attr.logger.info(f"Loaded policies {loaded} into env sampler from {path}")

    result = env_sampler.eval()
    env_sampler.post_evaluate(result["info"], -1)

    if isinstance(env_sampler, BatchEnvSampler):
        env_sampler.exit()


if __name__ == "__main__":
    scenario_path = env_str_helper(get_env("SCENARIO_PATH"))
    scenario_path = os.path.normpath(scenario_path)
    sys.path.insert(0, os.path.dirname(scenario_path))
    module = importlib.import_module(os.path.basename(scenario_path))

    main(getattr(module, "rl_component_bundle"), WorkflowEnvAttributes(), args=_get_args())
