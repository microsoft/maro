# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from maro.rl.workflows.helpers import from_env, get_logger, get_scenario_module

if __name__ == "__main__":
    mode = from_env("MODE")
    env_sampler = getattr(get_scenario_module(from_env("SCENARIODIR")), "get_env_sampler")()
    if mode == "sync":
        worker_id = from_env("WORKERID")
        logger = get_logger(
            from_env("LOGDIR", required=False, default=os.getcwd()), from_env("JOB"), f"ROLLOUT-WORKER.{worker_id}"
        )
        env_sampler.worker(
            from_env("ROLLOUTGROUP"), worker_id,
            proxy_kwargs={
                "redis_address": (from_env("REDISHOST"), from_env("REDISPORT")),
                "max_peer_discovery_retries": 50
            },
            logger=logger
        )
    elif mode == "async":
        num_episodes = from_env("NUMEPISODES")
        num_steps = from_env("NUMSTEPS", required=False, default=-1)
        actor_id = from_env("ACTORID")
        logger = get_logger(
            from_env("LOGDIR", required=False, default=os.getcwd()), from_env("JOB"), f"ACTOR.{actor_id}"
        )
        env_sampler.actor(
            from_env("GROUP"), actor_id, num_episodes,
            num_steps=num_steps,
            proxy_kwargs={
                "redis_address": (from_env("REDISHOST"), from_env("REDISPORT")),
                "max_peer_discovery_retries": 50
            },
            logger=logger
        )
    else:
        raise ValueError(f"MODE environment variable must be 'sync' or 'async', got {mode}")
