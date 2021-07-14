# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os.path import dirname, realpath

from maro.rl.learning import AgentWrapper
from maro.rl.policy import LocalPolicyManager

workflow_dir = dirname(dirname(realpath(__file__)))  # template directory
if workflow_dir not in sys.path:
    sys.path.insert(0, workflow_dir)

from general import agent2policy, log_dir, non_rl_policy_func_index, rl_policy_func_index, update_trigger, warmup


def get_agent_wrapper(mode: str = "inference-update"):
    assert mode in {"inference", "inference-update"}, f"mode must be 'inference' or 'inference-update', got {mode}"
    policy_dict = {
        **{name: func() for name, func in non_rl_policy_func_index.items()},
        **{name: func(mode=mode) for name, func in rl_policy_func_index.items()}
    }
    return AgentWrapper(
        LocalPolicyManager(
            policy_dict,
            update_trigger=update_trigger,
            warmup=warmup,
            log_dir=log_dir
        ),
        agent2policy
    )
