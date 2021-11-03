# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from maro.rl.workflows.helpers import from_env, get_log_dir, get_scenario_module

if __name__ == "__main__":
    mode = from_env("MODE")
    env_sampler = getattr(get_scenario_module(from_env("SCENARIODIR")), "get_env_sampler")()
    log_dir = get_log_dir(from_env("LOGDIR", required=False, default=os.getcwd()), from_env("JOB"))
    if mode == "sync":
        env_sampler.worker(
            from_env("ROLLOUTGROUP"), from_env("WORKERID"),
            proxy_kwargs={
                "redis_address": (from_env("REDISHOST"), from_env("REDISPORT")),
                "max_peer_discovery_retries": 50
            },
            log_dir=log_dir
        )
    elif mode == "async":
        num_episodes = from_env("NUMEPISODES")
        num_steps = from_env("NUMSTEPS", required=False, default=-1)
        env_sampler.actor(
            from_env("GROUP"), from_env("ACTORID"), num_episodes,
            num_steps=num_steps,
            proxy_kwargs={
                "redis_address": (from_env("REDISHOST"), from_env("REDISPORT")),
                "max_peer_discovery_retries": 50
            },
            log_dir=log_dir,
        )
    else:
        raise ValueError(f"MODE environment variable must be 'sync' or 'async', got {mode}")
