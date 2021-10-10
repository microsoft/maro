# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import importlib
import os
import sys

from maro.rl.workflows.helpers import from_env, get_default_log_dir

sys.path.insert(0, from_env("SCENARIODIR"))
module = importlib.import_module(from_env("SCENARIO"))
get_env_sampler = getattr(module, "get_env_sampler")

log_dir = from_env("LOGDIR", required=False, default=get_default_log_dir(from_env("JOB")))
os.makedirs(log_dir, exist_ok=True)


if __name__ == "__main__":
    mode = from_env("MODE")
    env_sampler = get_env_sampler()
    if mode == "sync":
        worker_id = from_env("WORKERID")
        env_sampler.worker(
            from_env("ROLLOUTGROUP"), worker_id,
            proxy_kwargs={
                "redis_address": (from_env("REDISHOST"), from_env("REDISPORT")),
                "max_peer_discovery_retries": 50
            },
            log_dir=log_dir
        )
    elif mode == "async":
        actor_id = from_env("ACTORID")
        num_episodes = from_env("NUMEPISODES")
        num_steps = from_env("NUMSTEPS", required=False, default=-1)
        env_sampler.actor(
            from_env("GROUP"), actor_id, num_episodes,
            num_steps=num_steps,
            proxy_kwargs={
                "redis_address": (from_env("REDISHOST"), from_env("REDISPORT")),
                "max_peer_discovery_retries": 50
            },
            log_dir=log_dir,
        )
    else:
        raise ValueError(f"MODE environment variable must be 'sync' or 'async', got {mode}")
