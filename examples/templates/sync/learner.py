# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import time
from os.path import dirname, realpath

from maro.rl.learning.sync import Learner, LocalRolloutManager, MultiNodeRolloutManager, MultiProcessRolloutManager

template_dir = dirname(dirname((realpath(__file__))))
if template_dir not in sys.path:
    sys.path.insert(0, template_dir)

from general import config, get_agent_wrapper, get_env_wrapper, log_dir
from policy_manager.policy_manager import get_policy_manager


def get_rollout_manager():
    rollout_mode = config["sync"]["rollout_mode"]
    if rollout_mode == "single-process":
        return LocalRolloutManager(
            get_env_wrapper(),
            get_agent_wrapper(),
            num_steps=config["num_steps"],
            log_dir=log_dir
        )
    if rollout_mode == "multi-process":
        return MultiProcessRolloutManager(
            config["sync"]["num_rollout_workers"],
            get_env_wrapper,
            get_agent_wrapper,
            num_steps=config["num_steps"],
            log_dir=log_dir,
        )
    if rollout_mode == "multi-node":
        return MultiNodeRolloutManager(
            config["sync"]["rollout_group"],
            config["sync"]["num_rollout_workers"],
            max_lag=config["max_lag"],
            proxy_kwargs={"redis_address": (config["redis"]["host"], config["redis"]["port"])}
        )

    raise ValueError(
        f"Unsupported roll-out mode: {rollout_mode}. Supported modes: single-process, multi-process, multi-node"
    )


if __name__ == "__main__":
    learner = Learner(
        policy_manager=get_policy_manager(),
        rollout_manager=get_rollout_manager(),
        num_episodes=config["num_episodes"],
        eval_schedule=config["eval_schedule"],
        log_dir=log_dir
    )
    time.sleep(10)
    learner.run()
