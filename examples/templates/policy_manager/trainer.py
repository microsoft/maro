# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
from os.path import dirname, realpath

from maro.rl import trainer_node

example_path = dirname(dirname(dirname(realpath(__file__))))  # DQN async mode directory
sys.path.insert(0, example_path)
from general import config, create_train_policy_func_index, log_dir


if __name__ == "__main__":
    trainer_node(
        config["policy_manager"]["group"],
        int(os.environ["TRAINERID"]),
        create_train_policy_func_index[config["scenario"]],
        proxy_kwargs={"redis_address": (config["redis"]["host"], config["redis"]["port"])},
        log_dir=log_dir
    )
