# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os.path import dirname, realpath

from maro.rl.learning import AgentWrapper

workflow_dir = dirname(dirname(realpath(__file__)))  # template directory
if workflow_dir not in sys.path:
    sys.path.insert(0, workflow_dir)

from general import agent2exploration, agent2policy, policy_func_index, exploration_func_index


def get_agent_wrapper():
    if exploration_func_index:
        exploration_dict = {name: func() for name, func in exploration_func_index.items()}
    else:
        exploration_dict = None

    return AgentWrapper(
        {name: func(learning=False) for name, func in policy_func_index.items()},
        agent2policy,
        exploration_dict=exploration_dict,
        agent2exploration=agent2exploration
    )
