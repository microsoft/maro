# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl.training import TrainOpsWorker
from maro.rl.utils.common import from_env, from_env_as_int, get_module
from maro.rl.workflows.utils import ScenarioAttr, _get_scenario_path
from maro.utils import Logger

if __name__ == "__main__":
    scenario = get_module(_get_scenario_path())
    scenario_attr = ScenarioAttr(scenario)
    
    worker_idx = from_env_as_int("ID")
    logger = Logger(
        f"TRAIN-WORKER.{worker_idx}", 
        dump_path=from_env("LOG_PATH"),
        dump_mode="a",
        stdout_level=from_env("LOG_LEVEL_STDOUT", required=False, default="CRITICAL"),
        file_level=from_env("LOG_LEVEL_FILE", required=False, default="CRITICAL")
    )
    worker = TrainOpsWorker(
        idx=from_env_as_int("ID"),
        policy_creator=scenario_attr.policy_creator,
        trainer_creator=scenario_attr.trainer_creator,
        producer_host=str(from_env("TRAIN_PROXY_HOST")),
        producer_port=from_env_as_int("TRAIN_PROXY_BACKEND_PORT"),
        logger=logger
    )
    worker.start()
