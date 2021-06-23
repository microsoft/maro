# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

from maro.rl import trainer_node

dqn_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # DQN async mode directory
sys.path.insert(0, dqn_path)
from general import AGENT_IDS, config, log_dir
from policy import get_independent_policy_for_training


if __name__ == "__main__":
    trainer_node(
        {id_: get_independent_policy_for_training for id_ in AGENT_IDS},
        config["policy_manager"]["group"],
        proxy_kwargs={
            "component_name": os.environ["TRAINERID"],
            "redis_address": (config["async"]["redis"]["host"], config["async"]["redis"]["port"])
        },
        log_dir=log_dir
    )
