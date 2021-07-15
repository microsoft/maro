# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import importlib
import sys
import yaml
from os.path import dirname, join, realpath

workflow_dir = dirname(realpath(__file__))
rl_example_dir = dirname(workflow_dir)

if rl_example_dir not in sys.path:
    sys.path.insert(0, rl_example_dir)

config_path = join(workflow_dir, "config.yml")
with open(config_path, "r") as config_file:
    config = yaml.safe_load(config_file)

log_dir = join(rl_example_dir, "log", config["job_name"])

module = importlib.import_module(f"{config['scenario']}")

agent2policy = getattr(module, "agent2policy")
get_env_wrapper = getattr(module, "get_env_wrapper")
non_rl_policy_func_index = getattr(module, "non_rl_policy_func_index", {})
rl_policy_func_index = getattr(module, "rl_policy_func_index")
update_trigger = getattr(module, "update_trigger")
warmup = getattr(module, "warmup")
post_collect = getattr(module, "post_collect", None)
post_evaluate = getattr(module, "end_of_evaluate", None)
post_update = getattr(module, "post_update", None)

num_rollouts = config["sync"]["num_rollout_workers"] if config["mode"] == "sync" else config["async"]["num_actors"]
replay_agents = [[] for _ in range(num_rollouts)]
for i, agent in enumerate(list(agent2policy.keys())):
    replay_agents[i % num_rollouts].append(agent)
