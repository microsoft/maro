# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os import getenv
from os.path import dirname, realpath

from maro.rl.learning import EnvironmentSampler


workflow_dir = dirname(dirname(realpath(__file__)))  # template directory
if workflow_dir not in sys.path:
    sys.path.insert(0, workflow_dir)

from agent_wrapper import get_agent_wrapper
from general import get_env_wrapper, get_eval_env_wrapper, log_dir


if __name__ == "__main__":
    index = getenv("WORKERID")
    if index is None:
        raise ValueError("Missing environment variable: WORKERID")
    index = int(index)

    env_sampler = EnvironmentSampler(get_env_wrapper, get_agent_wrapper, get_eval_env_wrapper=get_eval_env_wrapper)
    env_sampler.worker(
        getenv("ROLLOUTGROUP", default="rollout"), index,
        proxy_kwargs={
            "redis_address": (getenv("REDISHOST", default="maro-redis"), int(getenv("REDISPORT", default=6379))),
            "max_peer_discovery_retries": 50    
        },
        log_dir=log_dir
    )
