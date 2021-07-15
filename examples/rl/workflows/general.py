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

get_env_wrapper = getattr(module, "get_env_wrapper")
get_eval_env_wrapper = getattr(module, "get_eval_env_wrapper", lambda: None)
non_rl_policy_func_index = getattr(module, "non_rl_policy_func_index", {})
rl_policy_func_index = getattr(module, "rl_policy_func_index")
agent2policy = getattr(module, "agent2policy")
rl_agents = [agent_id for agent_id, policy_id in agent2policy.items() if policy_id in rl_policy_func_index]
update_trigger = getattr(module, "update_trigger")
warmup = getattr(module, "warmup")
post_collect = getattr(module, "post_collect", None)
post_evaluate = getattr(module, "post_evaluate", None)
post_update = getattr(module, "post_update", None)

# roll-out experience distribution amongst workers
num_rollouts = config["sync"]["num_rollout_workers"] if config["mode"] == "sync" else config["async"]["num_actors"]
if config["rollout_experience_distribution"]:
    replay_agents = [[] for _ in range(num_rollouts)]
    for i, agent in enumerate(rl_agents):
        replay_agents[i % num_rollouts].append(agent)
else:
    replay_agents = [rl_agents] * num_rollouts
