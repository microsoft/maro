# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from maro.rl_v3.utils.common import from_env, get_logger, get_module

if __name__ == "__main__":
    mode = from_env("MODE")
    env_sampler = getattr(get_module(str(from_env("SCENARIO_PATH"))), "get_env_sampler")()
    if mode == "sync":
        worker_id = from_env("WORKER_ID")
        logger = get_logger(
            str(from_env("LOG_PATH", required=False, default=os.getcwd())),
            str(from_env("JOB")),
            f"ROLLOUT-WORKER.{worker_id}",
        )
        env_sampler.worker(
            from_env("ROLLOUT_GROUP"), worker_id,
            proxy_kwargs={
                "redis_address": (from_env("REDIS_HOST"), from_env("REDIS_PORT")),
                "max_peer_discovery_retries": 50
            },
            logger=logger
        )
    elif mode == "async":
        num_episodes = from_env("NUM_EPISODES")
        num_steps = from_env("NUM_STEPS", required=False, default=-1)
        actor_id = from_env("ACTOR_ID")
        logger = get_logger(
            str(from_env("LOG_PATH", required=False, default=os.getcwd())),
            str(from_env("JOB")),
            f"ACTOR.{actor_id}"
        )
        env_sampler.actor(
            from_env("GROUP"), actor_id, num_episodes,
            num_steps=num_steps,
            proxy_kwargs={
                "redis_address": (from_env("REDIS_HOST"), from_env("REDIS_PORT")),
                "max_peer_discovery_retries": 50
            },
            logger=logger
        )
    else:
        raise ValueError(f"MODE environment variable must be 'sync' or 'async', got {mode}")
