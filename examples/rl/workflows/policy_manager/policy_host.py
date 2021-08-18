# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os import getenv
from os.path import dirname, realpath

from maro.rl.learning import policy_host

workflow_dir = dirname(dirname(realpath(__file__)))  # template directory
if workflow_dir not in sys.path:
    sys.path.insert(0, workflow_dir)

from general import log_dir, rl_policy_func_index


if __name__ == "__main__":
    host_id = getenv("HOSTID")
    if host_id is None:
        raise ValueError("missing environment variable: HOSTID")

    policy_host(
        rl_policy_func_index,
        int(host_id),
        getenv("LEARNGROUP", default="LEARN"),
        proxy_kwargs={
            "redis_address": (getenv("REDISHOST", default="maro-redis"), int(getenv("REDISPORT", default=6379))),
            "max_peer_discovery_retries": 50
        },
        log_dir=log_dir
    )
