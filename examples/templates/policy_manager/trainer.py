# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

from maro.rl import trainer_node

dqn_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # DQN async mode directory
sys.path.insert(0, dqn_path)
from general import config, create_policy_func_index, log_dir


if __name__ == "__main__":
    trainer_node(
        config["policy_manager"]["group"],
        int(os.environ["TRAINERID"]),
        create_policy_func_index[config["scenario"]],
        proxy_kwargs={"redis_address": (config["redis"]["host"], config["redis"]["port"])},
        log_dir=log_dir
    )
