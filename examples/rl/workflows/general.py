# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import importlib
import sys
from os import getenv
from os.path import dirname, join, realpath

workflow_dir = dirname(realpath(__file__))
rl_example_dir = dirname(workflow_dir)

if rl_example_dir not in sys.path:
    sys.path.insert(0, rl_example_dir)

log_dir = join(rl_example_dir, "log", getenv("JOB", ""))

module = importlib.import_module(f"{getenv('SCENARIO')}")

policy_func_dict = getattr(module, "policy_func_dict")
agent2policy = getattr(module, "agent2policy")
get_env_sampler = getattr(module, "get_env_sampler")
post_collect = getattr(module, "post_collect", None)
post_evaluate = getattr(module, "post_evaluate", None)
