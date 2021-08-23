# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os import getenv
from os.path import dirname, realpath

workflow_dir = dirname(dirname(realpath(__file__)))  # DQN directory
if workflow_dir not in sys.path:
    sys.path.insert(0, workflow_dir)

from policy_manager.policy_manager import get_policy_manager
from general import log_dir


if __name__ == "__main__":
    policy_manager = get_policy_manager()
    policy_manager.server(
        getenv("GROUP", default="ASYNC"),
        int(getenv("NUMROLLOUTS", default=5)),
        max_lag=int(getenv("MAXLAG", default=0)),
        proxy_kwargs={
            "redis_address": (getenv("REDISHOST", default="maro-redis"), int(getenv("REDISPORT", default=6379))),
            "max_peer_discovery_retries": 50    
        },
        log_dir=log_dir
    )
