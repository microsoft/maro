# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl.training import TrainOpsWorker
from maro.rl.utils.common import get_env, int_or_none
from maro.rl.workflows.scenario import Scenario
from maro.utils import LoggerV2

if __name__ == "__main__":
    scenario_attr = Scenario(get_env("SCENARIO_PATH"))
    worker_idx = int_or_none(get_env("ID"))
    logger = LoggerV2(
        f"TRAIN-WORKER.{worker_idx}",
        dump_path=get_env("LOG_PATH"),
        dump_mode="a",
        stdout_level=get_env("LOG_LEVEL_STDOUT", required=False, default="CRITICAL"),
        file_level=get_env("LOG_LEVEL_FILE", required=False, default="CRITICAL"),
    )
    worker = TrainOpsWorker(
        idx=int_or_none(get_env("ID")),
        policy_creator=scenario_attr.policy_creator,
        algorithm_instance_creator=scenario_attr.algorithm_instance_creator,
        producer_host=get_env("TRAIN_PROXY_HOST"),
        producer_port=int_or_none(get_env("TRAIN_PROXY_BACKEND_PORT")),
        logger=logger,
    )
    worker.start()
