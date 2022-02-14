# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.rl.rollout import RolloutWorker
from maro.rl.utils.common import get_env, int_or_none
from maro.rl.workflows.scenario import Scenario
from maro.utils import Logger

if __name__ == "__main__":
    scenario_attr = Scenario(get_env("SCENARIO_PATH"))
    policy_creator = scenario_attr.policy_creator

    worker_idx = int_or_none(get_env("ID"))
    logger = Logger(
        f"ROLLOUT-WORKER.{worker_idx}",
        dump_path=get_env("LOG_PATH"),
        dump_mode="a",
        stdout_level=get_env("LOG_LEVEL_STDOUT", required=False, default="CRITICAL"),
        file_level=get_env("LOG_LEVEL_FILE", required=False, default="CRITICAL"),
    )
    worker = RolloutWorker(
        idx=worker_idx,
        env_sampler_creator=lambda: scenario_attr.get_env_sampler(policy_creator),
        producer_host=get_env("ROLLOUT_CONTROLLER_HOST"),
        producer_port=int_or_none(get_env("ROLLOUT_CONTROLLER_PORT")),
        logger=logger,
    )
    worker.start()
