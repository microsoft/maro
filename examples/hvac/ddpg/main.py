# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from maro.utils import LogFormat, Logger

from examples.hvac.ddpg.callbacks import post_collect, post_evaluate
from examples.hvac.ddpg.config import training_config
from examples.hvac.ddpg.env_sampler import get_env_sampler


if __name__ == "__main__":
    os.makedirs(training_config["checkpoint_path"], exist_ok=True)
    os.makedirs(training_config["log_path"], exist_ok=True)
    logger = Logger(tag="MAIN", dump_folder=training_config["log_path"], format_=LogFormat.simple)

    env_sampler = get_env_sampler()
    for ep in range(training_config["num_episodes"]):
        while True:
            result = env_sampler.sample(return_rollout_info=False)
            # post_collect(result["tracker"], episode=ep, path=training_config["log_path"])
            logger.info(f"Ep {ep}: Collection finished")

            env_sampler.agent_wrapper.improve(checkpoint_dir=training_config["checkpoint_path"])

            if result["end_of_episode"]:
                break

        if (
            ep % training_config["evaluate_interval"] == 0
            or ep == training_config["num_episodes"] - 1
        ):
            trackers = env_sampler.test()
            post_evaluate(trackers, episode=ep, path=training_config["log_path"])
            logger.info(f"Ep {ep}: Evaluation finished")
