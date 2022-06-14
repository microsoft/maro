# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import importlib
import os
import sys
from typing import Type

from maro.rl.rl_component.rl_component_bundle import RLComponentBundle
from maro.rl.rollout import RolloutWorker
from maro.rl.utils.common import get_env, int_or_none
from maro.utils import LoggerV2

if __name__ == "__main__":
    scenario_path = get_env("SCENARIO_PATH")
    scenario_path = os.path.normpath(scenario_path)
    sys.path.insert(0, os.path.dirname(scenario_path))
    module = importlib.import_module(os.path.basename(scenario_path))

    rl_component_bundle_cls: Type[RLComponentBundle] = getattr(module, "rl_component_bundle_cls")
    rl_component_bundle = rl_component_bundle_cls()

    worker_idx = int_or_none(get_env("ID"))
    logger = LoggerV2(
        f"ROLLOUT-WORKER.{worker_idx}",
        dump_path=get_env("LOG_PATH"),
        dump_mode="a",
        stdout_level=get_env("LOG_LEVEL_STDOUT", required=False, default="CRITICAL"),
        file_level=get_env("LOG_LEVEL_FILE", required=False, default="CRITICAL"),
    )
    worker = RolloutWorker(
        idx=worker_idx,
        rl_component_bundle=rl_component_bundle,
        producer_host=get_env("ROLLOUT_CONTROLLER_HOST"),
        producer_port=int_or_none(get_env("ROLLOUT_CONTROLLER_PORT")),
        logger=logger,
    )
    worker.start()
