# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os import environ
from os.path import dirname, realpath

from maro.rl import rollout_worker_node

example_path = dirname(dirname(dirname(realpath(__file__))))  # example directory
sys.path.insert(0, example_path)
from general import config, get_agent_wrapper, get_env_wrapper, log_dir


if __name__ == "__main__":
    rollout_worker_node(
        config["rollout"]["group"],
        int(environ["WORKERID"]),
        get_env_wrapper(),
        get_agent_wrapper(),
        proxy_kwargs={"redis_address": (config["redis"]["host"], config["redis"]["port"])},
        log_dir=log_dir
    )
