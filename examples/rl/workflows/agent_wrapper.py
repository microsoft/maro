# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os.path import dirname, realpath

from maro.rl.learning import AgentWrapper

workflow_dir = dirname(dirname(realpath(__file__)))  # template directory
if workflow_dir not in sys.path:
    sys.path.insert(0, workflow_dir)

from general import agent2policy, non_rl_policy_func_index, rl_policy_func_index


def get_agent_wrapper():
    return AgentWrapper(
        {
            **{name: func() for name, func in non_rl_policy_func_index.items()},
            **{name: func(learning=False) for name, func in rl_policy_func_index.items()}
        },
        agent2policy
    )
