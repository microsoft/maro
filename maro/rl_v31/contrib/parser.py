# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import yaml


class RLConfigParser(object):
    def __init__(self, config_path: str, test_only: bool) -> None:
        self._raw_config = yaml.load(open(config_path, "r"), Loader=yaml.SafeLoader)
        self._test_only = test_only

    def parse(self) -> dict:
        config = {
            "job_name": self._raw_config["job"],
            "scenario_path": self._raw_config["scenario_path"],
            "log_path": self._raw_config["log_path"],
            "main": self._parse_main(),
            "rollout": self._parse_rollout(),
            "train": self._parse_train() if not self._test_only else {},
            "valid": self._parse_valid() if not self._test_only else {},
            "test": self._parse_test(),
        }

        return config

    def _parse_main(self) -> dict:
        return {
            "num_iterations": self._raw_config["main"]["num_iterations"],
            "explore_in_training": self._raw_config["main"].get("explore_in_training", True),
            "explore_in_valid_and_test": self._raw_config["main"].get("explore_in_valid_and_test", False),
        }

    def _parse_rollout(self) -> dict:
        rollout_config = self._raw_config.get("rollout", {})
        parallelism = rollout_config.get("parallelism", 1)
        parallel_type = rollout_config.get("parallel_type", "dummy")
        assert parallel_type in {"dummy"}

        return {
            "parallelism": parallelism,
            "parallel_type": parallel_type,
        }

    def _parse_train(self) -> dict:
        if "early_stop" in self._raw_config["train"] and self._raw_config["train"]["early_stop"] is not None:
            early_stop_config = (
                self._raw_config["train"]["early_stop"]["monitor"],
                self._raw_config["train"]["early_stop"]["patience"],
                self._raw_config["train"]["early_stop"].get("higher_better", True),
            )
        else:
            early_stop_config = None

        return {
            "steps_per_iteration": self._raw_config["train"].get("steps_per_iteration", None),
            "episodes_per_iteration": self._raw_config["train"].get("episodes_per_iteration", None),
            "checkpoint_path": self._raw_config["train"].get("checkpoint_path", None),
            "validation_interval": self._raw_config["train"].get("validation_interval", None),
            "load_path": self._raw_config["train"].get("load_path", None),
            "early_stop_config": early_stop_config,
        }

    def _parse_valid(self) -> dict:
        return {
            "steps_per_iteration": self._raw_config["valid"].get("steps_per_iteration", None),
            "episodes_per_iteration": self._raw_config["valid"].get("episodes_per_iteration", None),
        }

    def _parse_test(self) -> dict:
        return {
            "steps_per_iteration": self._raw_config["test"].get("steps_per_iteration", None),
            "episodes_per_iteration": self._raw_config["test"].get("episodes_per_iteration", None),
            "load_path": self._raw_config["test"].get("load_path", None),
        }
