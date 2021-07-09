# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
from os import environ
from os.path import dirname, realpath

from maro.rl.learning import AgentWrapper
from maro.rl.learning.synchronous import rollout_worker_node


workflow_dir = dirname(dirname(realpath(__file__)))  # template directory
if workflow_dir not in sys.path:
    sys.path.insert(0, workflow_dir)

from general import (
    agent2exploration, agent2policy, config, policy_func_index, exploration_func_index, get_env_wrapper, log_dir
)


if __name__ == "__main__":
    rollout_worker_node(
        config["sync"]["rollout_group"],
        int(environ["WORKERID"]),
        get_env_wrapper(),
        AgentWrapper(
            {name: func(learning=False) for name, func in policy_func_index.items()},
            agent2policy,
            exploration_dict={name: func() for name, func in exploration_func_index.items()},
            agent2exploration=agent2exploration
        ),
        proxy_kwargs={"redis_address": (config["redis"]["host"], config["redis"]["port"])},
        log_dir=log_dir
    )
