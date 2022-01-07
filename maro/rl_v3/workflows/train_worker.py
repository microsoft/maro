# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl_v3.distributed import Worker
from maro.rl_v3.training.ops_creator import OpsCreator
from maro.rl_v3.utils.common import from_env, get_module


scenario = get_module(from_env("SCENARIO_PATH"))
policy_creator = getattr(scenario, "policy_creator")
trainer_creator = getattr(scenario, "trainer_creator")
Worker("train", from_env("ID"), OpsCreator(policy_creator, trainer_creator), from_env("DISPATCHER_HOST")).start()
