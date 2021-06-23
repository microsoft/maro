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
        policy_manager,
        config["async"]["num_actors"],
        config["async"]["group"],
        proxy_kwargs={"redis_address": (config["async"]["redis"]["host"], config["async"]["redis"]["port"])},
        log_dir=log_dir
    )
