# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

from maro.rl import policy_server

dqn_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # DQN directory
sys.path.insert(0, dqn_path)
from general import config, log_dir
from policy_manager import policy_manager


if __name__ == "__main__":
    policy_server(
        config["async"]["group"],
        policy_manager,
        config["async"]["num_actors"],
        proxy_kwargs={"redis_address": (config["redis"]["host"], config["redis"]["port"])},
        log_dir=log_dir
    )
