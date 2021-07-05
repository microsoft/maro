# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os import environ
from os.path import dirname, realpath

from maro.rl.policy import trainer_node

template_dir = dirname(dirname(realpath(__file__)))  # template directory
if template_dir not in sys.path:
    sys.path.insert(0, template_dir)
from scenario_index import config, train_policy_func_index, log_dir


if __name__ == "__main__":
    trainer_node(
        config["policy_manager"]["train_group"],
        int(environ["TRAINERID"]),
        train_policy_func_index,
        proxy_kwargs={"redis_address": (config["redis"]["host"], config["redis"]["port"])},
        log_dir=log_dir
    )
