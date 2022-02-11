# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl.training import TrainOpsWorker
from maro.rl.utils.common import from_env, from_env_as_int
from maro.rl.workflows.scenario import Scenario
from maro.utils import Logger

if __name__ == "__main__":
    scenario_attr = Scenario(str(from_env("SCENARIO_PATH")))
    worker_idx = from_env_as_int("ID")
    logger = Logger(
        f"TRAIN-WORKER.{worker_idx}",
        dump_path=str(from_env("LOG_PATH")),
        dump_mode="a",
        stdout_level=str(from_env("LOG_LEVEL_STDOUT", required=False, default="CRITICAL")),
        file_level=str(from_env("LOG_LEVEL_FILE", required=False, default="CRITICAL")),
    )
    worker = TrainOpsWorker(
        idx=from_env_as_int("ID"),
        policy_creator=scenario_attr.policy_creator,
        trainer_creator=scenario_attr.trainer_creator,
        producer_host=str(from_env("TRAIN_PROXY_HOST")),
        producer_port=from_env_as_int("TRAIN_PROXY_BACKEND_PORT"),
        logger=logger,
    )
    worker.start()
