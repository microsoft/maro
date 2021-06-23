# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

from maro.rl import policy_server

async_mode_path = os.path.dirname(os.path.realpath(__file__))  # DQN async mode directory
dqn_path = os.path.dirname(async_mode_path)  # DQN directory
cim_path = os.path.dirname(dqn_path)  # CIM example directory
sys.path.insert(0, cim_path)
sys.path.insert(0, dqn_path)
sys.path.insert(0, async_mode_path)
from general import config, log_dir
from policy_manager import policy_manager


if __name__ == "__main__":
    policy_server(
        policy_manager,
        config["async"]["num_actors"],
        config["async"]["group"],
        proxy_kwargs={"redis_address": (config["redis"]["host"], config["redis"]["port"])},
        log_dir=log_dir
    )
