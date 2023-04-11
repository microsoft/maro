# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import argparse
import importlib
import os
import sys
from typing import Optional

from maro.rl_v31.contrib.parser import RLConfigParser
from maro.rl_v31.workflow.workflow import Workflow
from maro.utils import set_seeds


def run(config: dict, test_only: bool = False, seed: Optional[int] = None) -> None:
    if seed is not None:
        set_seeds(seed)

    scenario_path = config["scenario_path"]
    scenario_path = os.path.normpath(scenario_path)
    sys.path.insert(0, os.path.dirname(scenario_path))
    module = importlib.import_module(os.path.basename(scenario_path))

    rcb = getattr(module, "rl_component_bundle")
    workflow = Workflow(
        rl_component_bundle=rcb,
        rollout_parallelism=config["rollout"]["parallelism"],  # TODO: add parallel_type
        log_path=config["log_path"],
    )

    if not test_only:
        workflow.train(
            num_iterations=config["main"]["num_iterations"],
            steps_per_iteration=config["train"]["steps_per_iteration"],
            episodes_per_iteration=config["train"]["episodes_per_iteration"],
            valid_steps_per_iteration=config["valid"]["steps_per_iteration"],
            valid_episodes_per_iteration=config["valid"]["episodes_per_iteration"],
            checkpoint_path=config["train"]["checkpoint_path"],
            validation_interval=config["train"]["validation_interval"],
            explore_in_training=config["main"]["explore_in_training"],
            explore_in_validation=config["main"]["explore_in_valid_and_test"],
            early_stop_config=config["train"]["early_stop_config"],
            load_path=config["train"]["load_path"],
        )

    workflow.test(
        save_path=config["log_path"],
        steps_per_iteration=config["test"]["steps_per_iteration"],
        episodes_per_iteration=config["test"]["episodes_per_iteration"],
        explore=config["main"]["explore_in_valid_and_test"],
        load_path=config["test"]["load_path"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MARO RL workflow parser")
    parser.add_argument("--config", help="Path of the job deployment")
    parser.add_argument("--seed", type=int, required=False, help="The random seed set before running this job")
    parser.add_argument("--test_only", action="store_true", help="Only run evaluation part of the workflow")

    args = parser.parse_args()

    parser = RLConfigParser(args.config, args.test_only)
    config = parser.parse()

    run(config, test_only=args.test_only, seed=args.seed)
