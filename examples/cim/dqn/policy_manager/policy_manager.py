# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys

from maro.rl import LocalPolicyManager, MultiNodePolicyManager, MultiProcessPolicyManager

dqn_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # DQN directory
sys.path.insert(0, dqn_path)
from general import AGENT_IDS, NUM_POLICY_TRAINERS, config, log_dir
from policy import get_independent_policy_for_training

def get_policy_manager():
    policies = [get_independent_policy_for_training(i) for i in AGENT_IDS]
    training_mode = config["policy_manager"]["policy_training_mode"]
    if training_mode == "single-process":
        return LocalPolicyManager(policies, log_dir=log_dir)
    if training_mode == "multi-process":
        return MultiProcessPolicyManager(
            policies,
            {id_: f"TRAINER.{id_ % NUM_POLICY_TRAINERS}" for id_ in AGENT_IDS}, # policy-trainer mapping
            {i: get_independent_policy_for_training for i in AGENT_IDS},
            log_dir=log_dir
        )
    if training_mode == "multi-node":
        return MultiNodePolicyManager(
            config["policy_manager"]["group"],
            policies,
            {id_: f"TRAINER.{id_ % NUM_POLICY_TRAINERS}" for id_ in AGENT_IDS}, # policy-trainer mapping
            proxy_kwargs={"redis_address": (config["redis"]["host"], config["redis"]["port"])},
            log_dir=log_dir
        )

    raise ValueError(
        f"Unsupported policy training mode: {training_mode}. "
        f"Supported modes: single-process, multi-process, multi-node"
    )
