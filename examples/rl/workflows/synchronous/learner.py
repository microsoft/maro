# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os.path import dirname, realpath

from maro.rl.learning.synchronous import (
    Learner, LocalRolloutManager, MultiNodeRolloutManager, MultiProcessRolloutManager
)

workflow_dir = dirname(dirname((realpath(__file__))))
if workflow_dir not in sys.path:
    sys.path.insert(0, workflow_dir)

from agent_wrapper import get_agent_wrapper
from policy_manager.policy_manager import get_policy_manager
from general import config, post_collect, post_evaluate, get_env_wrapper, log_dir


def get_rollout_manager():
    rollout_mode = config["sync"]["rollout_mode"]
    if rollout_mode == "single-process":
        return LocalRolloutManager(
            get_env_wrapper(),
            get_agent_wrapper(),
            num_steps=config["num_steps"],
            post_collect=post_collect,
            post_evaluate=post_evaluate,
            log_dir=log_dir
        )
    if rollout_mode == "multi-process":
        return MultiProcessRolloutManager(
            config["sync"]["num_rollout_workers"],
            get_env_wrapper,
            get_agent_wrapper,
            num_steps=config["num_steps"],
            post_collect=post_collect,
            post_evaluate=post_evaluate,
            log_dir=log_dir,
        )
    if rollout_mode == "multi-node":
        return MultiNodeRolloutManager(
            config["sync"]["rollout_group"],
            config["sync"]["num_rollout_workers"],
            num_steps=config["num_steps"],
            max_lag=config["max_lag"],
            min_finished_workers=config["sync"]["min_finished_workers"],
            max_extra_recv_tries=config["sync"]["max_extra_recv_tries"],
            extra_recv_timeout=config["sync"]["extra_recv_timeout"],
            post_collect=post_collect,
            post_evaluate=post_evaluate,
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
    learner.run()
