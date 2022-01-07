# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl_v3.distributed import Worker
from maro.rl_v3.utils.common import from_env, get_module


scenario = get_module(from_env("SCENARIO_PATH"))
env_sampler_creator = getattr(scenario, "env_sampler_creator")
Worker("rollout", from_env("ID"), env_sampler_creator, "127.0.0.1").start()
