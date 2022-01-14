# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl_v3.distributed import Worker
from maro.rl_v3.training.ops_creator import OpsCreator
from maro.rl_v3.utils.common import from_env, from_env_as_int, get_module

from .utils import ScenarioAttr, _get_scenario_path

scenario = get_module(_get_scenario_path())
scenario_attr = ScenarioAttr(scenario)

worker = Worker(
    "train",
    from_env_as_int("ID"),
    OpsCreator(scenario_attr.policy_creator, scenario_attr.trainer_creator),
    str(from_env("DISPATCHER_HOST")),
    router_port=from_env_as_int("DISPATCHER_BACKEND_PORT")
)

worker.start()
