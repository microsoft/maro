# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os

import numpy as np
from shutil import copy2

from maro.utils import LogFormat, Logger, set_seeds

from examples.hvac.rl.callbacks import post_evaluate, visualize_returns
from examples.hvac.rl.config import Config
from examples.hvac.rl.env_sampler import get_env_sampler


def train(config: Config, model_path: str):
    copy2(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py"), config.log_dir)
    logger = Logger(tag="Train", dump_folder=config.log_dir, format_=LogFormat.simple)
    logger.debug(config)

    env_sampler = get_env_sampler()

    if model_path:
        env_sampler.agent_wrapper.load(model_path)
        logger.info(f"Load model state from {model_path}")

    total_return, total_return_test = [], []
    rolling_total_return = []
    rolling_window = 10

    for ep in range(config.num_episode):
        while True:
            result = env_sampler.sample(return_rollout_info=False)
            logger.info(f"Ep {ep}: Collection finished")

            env_sampler.agent_wrapper.improve(config.checkpoint_dir, ep if (ep % 20 == 19) else None)

            if result["end_of_episode"]:
                break

        tracker = result["tracker"]
        total_return.append(tracker["total_reward"][-1])
        rolling_total_return.append(np.mean(total_return[-rolling_window:]))

        if ep % config.evaluate_interval == 0 or ep == config.num_episode - 1:
            trackers = env_sampler.test()
            res = post_evaluate(trackers, episode=ep, path=config.log_dir, prefix="Eval")
            total_return_test.append(trackers["total_reward"][-1])
            logger.info(f"Ep {ep}: Evaluation Improvement {res}")

    visualize_returns(total_return, config.log_dir, "Return")
    visualize_returns(rolling_total_return, config.log_dir, "Rolling Return")
    visualize_returns(total_return_test, config.log_dir, f"Test Return per {config.evaluate_interval} eps")


def test(config: Config, model_path: str):
    logger = Logger(tag="Test", dump_folder=config.log_dir, format_=LogFormat.simple)

    env_sampler = get_env_sampler()
    env_sampler.agent_wrapper.load(model_path, is_file=True)
    logger.info(f"Load model state from {model_path}")

    tracker = env_sampler.test()
    logger.info(f"Exploitation finished")

    res = post_evaluate(
        tracker, episode=-1, path=config.log_dir,
        prefix="Eval" if env_sampler.agent_wrapper.exploit_mode else "Train"
    )
    logger.info(f"Final improvement: {res}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action='store_true', default=False)
    parser.add_argument("--file", "-f", dest="model_path", type=str, default=None)
    args = parser.parse_args()

    config = Config()
    set_seeds(config.seed)
    if args.eval:
        test(config, args.model_path)
    else:
        train(config, args.model_path)
