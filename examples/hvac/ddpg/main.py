# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from maro.utils import LogFormat, Logger

from examples.hvac.ddpg.callbacks import post_collect, post_evaluate
from examples.hvac.ddpg.config import experiment_name, training_config
from examples.hvac.ddpg.env_sampler import get_env_sampler

checkpoint_dir = os.path.join(training_config["checkpoint_path"], experiment_name)
os.makedirs(checkpoint_dir, exist_ok=True)

log_dir = os.path.join(training_config["log_path"], experiment_name)
os.makedirs(log_dir, exist_ok=True)


def train():
    logger = Logger(tag="Train", dump_folder=log_dir, format_=LogFormat.simple)

    env_sampler = get_env_sampler()

    if training_config["load_model"]:
        env_sampler.agent_wrapper.load(checkpoint_dir)
        logger.info(f"Load model state from {checkpoint_dir}")

    for ep in range(training_config["num_episodes"]):
        while True:
            result = env_sampler.sample(return_rollout_info=False)
            logger.info(f"Ep {ep}: Collection finished")

            env_sampler.agent_wrapper.improve(checkpoint_dir=checkpoint_dir)

            if result["end_of_episode"]:
                break

        if (
            ep % training_config["evaluate_interval"] == 0
            or ep == training_config["num_episodes"] - 1
        ):
            trackers = env_sampler.test()
            post_evaluate(trackers, episode=ep, path=log_dir, prefix="Train")   # Train due to the implementation
            logger.info(f"Ep {ep}: Evaluation finished")


def test():
    logger = Logger(tag="Test", dump_folder=log_dir, format_=LogFormat.simple)

    env_sampler = get_env_sampler()

    if training_config["load_model"]:
        env_sampler.agent_wrapper.load(checkpoint_dir)
        logger.info(f"Load model state from {checkpoint_dir}")

    env_sampler.agent_wrapper.exploit()

    result = env_sampler.sample(return_rollout_info=False)
    logger.info(f"Exploitation finished")

    post_evaluate(result["tracker"], episode=-1, path=log_dir, prefix="Eval" if env_sampler.agent_wrapper.exploit_mode else "Train")


if __name__ == "__main__":
    if training_config["test"]:
        test()
    else:
        train()
