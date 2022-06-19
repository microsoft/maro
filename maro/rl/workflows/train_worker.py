# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import importlib
import os
import sys

from maro.rl.rl_component.rl_component_bundle import RLComponentBundle
from maro.rl.training import TrainOpsWorker
from maro.rl.utils.common import get_env, int_or_none
from maro.rl.workflows.utils import env_str_helper
from maro.utils import LoggerV2

if __name__ == "__main__":
    scenario_path = env_str_helper(get_env("SCENARIO_PATH"))
    scenario_path = os.path.normpath(scenario_path)
    sys.path.insert(0, os.path.dirname(scenario_path))
    module = importlib.import_module(os.path.basename(scenario_path))

    rl_component_bundle: RLComponentBundle = getattr(module, "rl_component_bundle")

    worker_idx = int_or_none(get_env("ID"))
    logger = LoggerV2(
        f"TRAIN-WORKER.{worker_idx}",
        dump_path=get_env("LOG_PATH"),
        dump_mode="a",
        stdout_level=get_env("LOG_LEVEL_STDOUT", required=False, default="CRITICAL"),
        file_level=get_env("LOG_LEVEL_FILE", required=False, default="CRITICAL"),
    )
    worker = TrainOpsWorker(
        idx=int(env_str_helper(get_env("ID"))),
        rl_component_bundle=rl_component_bundle,
        producer_host=env_str_helper(get_env("TRAIN_PROXY_HOST")),
        producer_port=int_or_none(get_env("TRAIN_PROXY_BACKEND_PORT")),
        logger=logger,
    )
    worker.start()
