# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl.rollout import RolloutWorker
from maro.rl.utils.common import from_env, from_env_as_int, get_module
from maro.rl.workflows.utils import ScenarioAttr, _get_scenario_path
from maro.utils import Logger

if __name__ == "__main__":
    scenario = get_module(_get_scenario_path())
    scenario_attr = ScenarioAttr(scenario)
    policy_creator = scenario_attr.policy_creator

    worker_idx = from_env_as_int("ID")
    logger = Logger(
        f"ROLLOUT-WORKER.{worker_idx}", 
        dump_path=from_env("LOG_PATH"),
        dump_mode="a",
        stdout_level=from_env("LOG_LEVEL_STDOUT", required=False, default="CRITICAL"),
        file_level=from_env("LOG_LEVEL_FILE", required=False, default="CRITICAL")
    )
    worker = RolloutWorker(
        idx=worker_idx,
        env_sampler_creator=lambda: scenario_attr.get_env_sampler(policy_creator),
        proxy_host=str(from_env("ROLLOUT_PROXY_HOST")),
        proxy_port=from_env_as_int("ROLLOUT_PROXY_BACKEND_PORT"),
        logger=logger
    )
    worker.start()
