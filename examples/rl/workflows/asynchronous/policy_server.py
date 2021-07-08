# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os.path import dirname, realpath

from maro.rl.learning.asynchronous import policy_server

workflow_dir = dirname(dirname(realpath(__file__)))  # DQN directory
if workflow_dir not in sys.path:
    sys.path.insert(0, workflow_dir)

from policy_manager.policy_manager import get_policy_manager
from general import config, log_dir


if __name__ == "__main__":
    policy_server(
        config["async"]["group"],
        get_policy_manager(),
        config["async"]["num_actors"],
        max_lag=config["max_lag"],
        proxy_kwargs={"redis_address": (config["redis"]["host"], config["redis"]["port"])},
        log_dir=log_dir
    )
