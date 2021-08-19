# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os import getenv
from os.path import dirname, realpath

from maro.rl.learning.synchronous import rollout_worker_node


workflow_dir = dirname(dirname(realpath(__file__)))  # template directory
if workflow_dir not in sys.path:
    sys.path.insert(0, workflow_dir)

from agent_wrapper import get_agent_wrapper
from general import get_env_wrapper, get_eval_env_wrapper, log_dir, replay_agents


if __name__ == "__main__":
    worker_id = getenv("WORKERID")
    if worker_id is None:
        raise ValueError("Missing environment variable: WORKERID")
    worker_id = int(worker_id)

    rollout_worker_node(
        getenv("ROLLOUTGROUP", default="rollout"),
        worker_id,
        get_env_wrapper(replay_agent_ids=replay_agents[worker_id]),
        get_agent_wrapper(),
        eval_env_wrapper=get_eval_env_wrapper(),
        proxy_kwargs={
            "redis_address": (getenv("REDISHOST", default="maro-redis"), int(getenv("REDISPORT", default=6379))),
            "max_peer_discovery_retries": 50    
        },
        log_dir=log_dir
    )
