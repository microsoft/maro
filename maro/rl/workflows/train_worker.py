# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl.training import TrainOpsWorker
from maro.rl.utils.common import from_env, from_env_as_int, get_module
from maro.rl.workflows.utils import ScenarioAttr, _get_scenario_path

if __name__ == "__main__":
    scenario = get_module(_get_scenario_path())
    scenario_attr = ScenarioAttr(scenario)
    worker = TrainOpsWorker(
        from_env_as_int("ID"),
        scenario_attr.policy_creator,
        scenario_attr.trainer_creator,
        str(from_env("DISPATCHER_HOST")),
        router_port=from_env_as_int("DISPATCHER_BACKEND_PORT")
    )
    worker.start()
