# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ipaddress
import os
from typing import Union

import yaml

from maro.utils.logger import LEVEL_MAP


class ConfigParser:
    """Configuration parser for running RL workflows.

    Args:
        config (Union[str, dict]): A dictionary configuration or a path to a Yaml file that contains
            the configuration. If it is a path, the parser will attempt to read it into a dictionary
            in memory.
    """
    def __init__(self, config: Union[str, dict]) -> None:
        assert isinstance(config, (dict, str))
        if isinstance(config, str):
            with open(config, "r") as fp:
                self._config = yaml.safe_load(fp)
        else:
            self._config = config

        self._validation_err_pfx = f"Invalid configuration: {self._config}"
        self._validate()

    @property
    def config(self) -> dict:
        return self._config

    def _validate(self):
        if "job" not in self._config:
            raise KeyError(f"{self._validation_err_pfx}: missing field 'job'")
        if "scenario_path" not in self._config:
            raise KeyError(f"{self._validation_err_pfx}: missing field 'scenario_path'")
        if "log_path" not in self._config:
            raise KeyError(f"{self._validation_err_pfx}: missing field 'log_path'")

        self._validate_main_section()
        self._validate_rollout_section()
        self._validate_training_section()

    def _validate_main_section(self):
        if "main" not in self._config:
            raise KeyError(f"{self._validation_err_pfx}: missing field 'main'")

        if "num_episodes" not in self._config["main"]:
            raise KeyError(f"{self._validation_err_pfx}: missing field 'num_episodes' under section 'main'")

        num_episodes = self._config["main"]["num_episodes"]
        if not isinstance(num_episodes, int) or num_episodes < 1:
            raise ValueError(
                f"{self._validation_err_pfx}: field 'num_episodes' under section 'main' must be a positive int"
            )

        if "num_steps" in self._config["main"]:
            num_steps = self._config["main"]["num_steps"]
            if not isinstance(num_steps, int) or num_steps < -1 or num_steps == 0:
                raise ValueError(
                    f"{self._validation_err_pfx}: field 'num_steps' under section 'main' must be -1 or a positive int"
                )

        if "eval_schedule" in self._config["main"]:
            eval_schedule = self._config["main"]["eval_schedule"]
            if (
                not isinstance(eval_schedule, (int, list)) or
                isinstance(eval_schedule, int) and eval_schedule < 1 or
                isinstance(eval_schedule, list) and any(not isinstance(val, int) or val < 1 for val in eval_schedule)
            ):
                raise ValueError(
                    f"{self._validation_err_pfx}: field 'eval_schedule' under section 'main'"
                    f"must be a positive int or a list of positive ints"
                )

        if "logging" in self._config["main"]:
            self._validate_logging_section("main", self._config["main"]["logging"])

    def _validate_rollout_section(self):
        if "rollout" not in self._config or not isinstance(self._config["rollout"], dict):
            raise KeyError(f"{self._validation_err_pfx}: missing section 'rollout'")  

        # validate parallel rollout config
        if "parallelism" in self._config["rollout"]:
            parallelism = self._config["rollout"]["parallelism"]
            if not isinstance(parallelism, int):
                raise TypeError(
                    f"{self._validation_err_pfx}: field 'parallelism' under section 'rollout' must be an int"
                )
            if parallelism > 1:
                if "proxy" not in self._config["rollout"]:
                    raise KeyError(f"{self._validation_err_pfx}: missing field 'proxy' under section 'rollout'")
                self._validate_proxy_section(self._config["rollout"]["proxy"])

                # validate optional fields: eval_parallelism, min_env_samples, grace_factor
                if "eval_parallelism" in self._config["rollout"]:
                    eval_parallelism = self._config["rollout"]["eval_parallelism"]
                    if not isinstance(eval_parallelism, int) or eval_parallelism > parallelism:
                        raise ValueError(
                            f"{self._validation_err_pfx}: 'eval_parallelism' under section 'rollout' must be an int "
                            f"that does not exceed the value of field 'parallelism': {parallelism}"
                        )

                if "min_env_samples" in self._config["rollout"]:
                    min_env_samples = self._config["rollout"]["min_env_samples"]
                    if not isinstance(min_env_samples, int) or min_env_samples > parallelism:
                        raise ValueError(
                            f"{self._validation_err_pfx}: 'min_env_samples' under section 'rollout' must be an int "
                            f"that does not exceed the value of field 'parallelism': {parallelism}"
                        )

                if "grace_factor" in self._config["rollout"] and not isinstance(min_env_samples, (int, float)):
                    raise ValueError(
                        f"{self._validation_err_pfx}: 'grace_factor' under section 'rollout' must be an int or float"
                    )

                if "logging" in self._config["rollout"]:
                    self._validate_logging_section("rollout", self._config["rollout"]["logging"])

    def _validate_training_section(self):
        if "training" not in self._config or not isinstance(self._config["training"], dict):
            raise KeyError(f"{self._validation_err_pfx}: missing field 'training'")
        if "mode" not in self._config["training"]:
            raise KeyError(f"{self._validation_err_pfx}: missing field 'mode' under section 'training'")
        if self._config["training"]["mode"] not in {"simple", "parallel"}:
            raise ValueError(
                f"'mode' value under section 'training' must be 'simple' or 'parallel', got {self._config['mode']}"
            )

        if self._config["training"]["mode"] == "parallel":
            if "num_workers" not in self._config["training"]:
                raise KeyError(f"{self._validation_err_pfx}: missing field 'num_workers' under section 'training'")
            if "proxy" not in self._config["training"]:
                raise KeyError(f"{self._validation_err_pfx}: missing field 'proxy' under section 'training'")
            self._validate_proxy_section(self._config["training"]["proxy"])
            if "logging" in self._config["training"]:
                self._validate_logging_section("training", self._config["training"]["logging"])

        if "load_path" in self._config["training"] and not isinstance(self._config["training"]["load_path"], str):
            raise TypeError(f"{self._validation_err_pfx}: field 'load_path' must be a string")

        if "checkpointing" in self._config["training"]:
            self._validate_checkpointing_section(self._config["training"]["checkpointing"])

    def _validate_proxy_section(self, proxy_section: dict) -> None:
        if "host" not in proxy_section:
            raise KeyError(f"{self._validation_err_pfx}: missing field 'host' under section 'proxy'")
        if not isinstance(proxy_section["host"], str):
            raise TypeError(f"{self._validation_err_pfx}: field 'host' must be a string")
        # Check that the host string is a valid IP address
        try:
            ipaddress.ip_address(proxy_section["host"])
        except ValueError:
            raise ValueError(f"{self._validation_err_pfx}: field 'host' is not a valid IP address") 

        if "frontend" not in proxy_section:
            raise KeyError(f"{self._validation_err_pfx}: missing field 'frontend' under section 'proxy'")
        if not isinstance(proxy_section["frontend"], int):
            raise TypeError(f"{self._validation_err_pfx}: field 'frontend' must be an int")

        if "backend" not in proxy_section:
            raise KeyError(f"{self._validation_err_pfx}: missing field 'backend' under section 'proxy'")
        if not isinstance(proxy_section["backend"], int):
            raise TypeError(f"{self._validation_err_pfx}: field 'backend' must be an int")

    def _validate_checkpointing_section(self, section: dict) -> None:
        if "path" not in section:
            raise KeyError(f"{self._validation_err_pfx}: missing field 'path' under section 'checkpointing'")
        if not isinstance(section["path"], str):
            raise TypeError(f"{self._validation_err_pfx}: field 'path' under section 'checkpointing' must be a string")

        if "interval" in section:
            if not isinstance(section["interval"], int):
                raise TypeError(
                    f"{self._validation_err_pfx}: field 'interval' under section 'checkpointing' must be an int"
                )

    def _validate_logging_section(self, component, level_dict: dict) -> None:
        if any(key not in {"stdout", "file"} for key in level_dict):
            raise KeyError(
                f"{self._validation_err_pfx}: fields under section '{component}.logging' must be 'stdout' or 'file'"
            )
        valid_loggings = set(LEVEL_MAP.keys())
        for key, val in level_dict.items():
            if val not in valid_loggings:
                raise ValueError(
                    f"{self._validation_err_pfx}: '{component}.logging.{key}' must be one of {valid_loggings}."
                )

    def get_path_mapping(self, containerize: bool = False) -> dict:
        """Generate path mappings for a local or containerized environment.

        Args:
            containerize (bool): If true, the paths you specify in the configuration file (which should always be local)
                are mapped to paths inside the containers as follows:
                    local/scenario/path -> "/scenario"
                    local/load/path -> "loadpoint"
                    local/checkpoint/path -> "checkpoints"
                    local/log/path -> "/logs"
                Defaults to False.
        """
        path_map = {self._config["scenario_path"]: "/scenario" if containerize else self._config["scenario_path"]}
        path_map[self._config["log_path"]] = "/logs" if containerize else self._config["log_path"]
        if "load_path" in self._config["training"]:
            load_path = self._config["training"]["load_path"]
            path_map[load_path] = "/loadpoint" if containerize else load_path
        if "checkpointing" in self._config["training"]:
            ckpt_path = self._config["training"]["checkpointing"]["path"]
            path_map[ckpt_path] = "/checkpoints" if containerize else ckpt_path

        return path_map

    def to_env(self, containerize: bool = False) -> dict:
        """Generate environment variables for the workflow scripts.
        
        Args:
            containerize (bool): If true, the generated environment variables are to be used in a containerized
                environment. Only path-related environment variables are affected by this flag. See the docstring
                for ``get_path_mappings`` for details. Defaults to False.
        """
        path_mapping = self.get_path_mapping(containerize=containerize)
        parallelism = self._config["rollout"].get("parallelism", 1)
        env = {
            "main": {
                "JOB": self._config["job"],
                "NUM_EPISODES": str(self._config["main"]["num_episodes"]),
                "ROLLOUT_PARALLELISM": str(parallelism),
                "TRAIN_MODE": self._config["training"]["mode"],
                "SCENARIO_PATH": path_mapping[self._config["scenario_path"]]
            }
        }

        if "eval_schedule" in self._config["main"]:
            env["main"]["EVAL_SCHEDULE"] = str(self._config["main"]["eval_schedule"])

        if "load_path" in self._config["training"]:
            env["main"]["LOAD_PATH"] = path_mapping[self._config["training"]["load_path"]]

        if "checkpointing" in self._config["training"]:
            conf = self._config["training"]["checkpointing"]
            env["main"]["CHECKPOINT_PATH"] = path_mapping[conf["path"]]
            if "interval" in conf:
                env["main"]["CHECKPOINT_INTERVAL"] = str(conf["interval"])

        if "num_steps" in self._config["main"]:
            env["main"]["NUM_STEPS"] = str(self._config["main"]["num_steps"])

        if "logging" in self._config["main"]:
            env["main"].update({
                "LOG_LEVEL_STDOUT": self.config["main"]["logging"]["stdout"],
                "LOG_LEVEL_FILE": self.config["main"]["logging"]["file"]
            })

        if parallelism > 1:
            proxy_host = self._get_rollout_proxy_host(containerize=containerize)
            proxy_frontend_port = str(self._config["rollout"]["proxy"]["frontend"])
            proxy_backend_port = str(self._config["rollout"]["proxy"]["backend"])
            num_rollout_workers = self._config["rollout"]["parallelism"]
            env["main"].update({
                "ROLLOUT_PROXY_HOST": proxy_host,
                "ROLLOUT_PROXY_FRONTEND_PORT": str(proxy_frontend_port)
            })

            # optional settings for parallel rollout
            if "eval_parallelism" in self._config["rollout"]:
                env["main"]["EVAL_PARALLELISM"] = str(self._config["rollout"]["eval_parallelism"])
            if "min_env_samples" in self._config["rollout"]:
                env["main"]["MIN_ENV_SAMPLES"] = str(self._config["rollout"]["min_env_samples"])
            if "grace_factor" in self._config["rollout"]:
                env["main"]["GRACE_FACTOR"] = str(self._config["rollout"]["grace_factor"])

            env["rollout_proxy"] = {
                "NUM_ROLLOUT_WORKERS": str(num_rollout_workers),
                "ROLLOUT_PROXY_FRONTEND_PORT": proxy_frontend_port,
                "ROLLOUT_PROXY_BACKEND_PORT": proxy_backend_port
            }
            for i in range(num_rollout_workers):
                worker_id = f"rollout_worker-{i}"
                env[worker_id] = {
                    "ID": str(i),
                    "ROLLOUT_PROXY_HOST": proxy_host,
                    "ROLLOUT_PROXY_BACKEND_PORT": proxy_backend_port,
                    "SCENARIO_PATH": path_mapping[self._config["scenario_path"]]
                }
                if "logging" in self._config["rollout"]:
                    env[worker_id].update({
                        "LOG_LEVEL_STDOUT": self.config["rollout"]["logging"]["stdout"],
                        "LOG_LEVEL_FILE": self.config["rollout"]["logging"]["file"]
                    })

        if self._config["training"]["mode"] == "parallel":
            conf = self._config['training']['proxy']
            proxy_host = self._get_train_proxy_host(containerize=containerize)
            proxy_frontend_port = str(conf["frontend"])
            proxy_backend_port = str(conf["backend"])
            num_workers = self._config["training"]["num_workers"]
            env["main"].update({
                "TRAIN_PROXY_HOST": proxy_host, "TRAIN_PROXY_FRONTEND_PORT": proxy_frontend_port
            })
            env["train_proxy"] = {
                "TRAIN_PROXY_FRONTEND_PORT": proxy_frontend_port,
                "TRAIN_PROXY_BACKEND_PORT": proxy_backend_port
            }
            for i in range(num_workers):
                worker_id = f"train_worker-{i}"
                env[worker_id] = {
                    "ID": str(i),
                    "TRAIN_PROXY_HOST": proxy_host,
                    "TRAIN_PROXY_BACKEND_PORT": proxy_backend_port,
                    "SCENARIO_PATH": path_mapping[self._config["scenario_path"]]
                }
                if "logging" in self._config["training"]:
                    env[worker_id].update({
                        "LOG_LEVEL_STDOUT": self.config["training"]["logging"]["stdout"],
                        "LOG_LEVEL_FILE": self.config["training"]["logging"]["file"]
                    })

        # All components write logs to the same file
        for vars in env.values():
            vars["LOG_PATH"] = os.path.join(path_mapping[self._config["log_path"]], self._config["job"])

        return env

    def _get_rollout_proxy_host(self, containerize: bool = False):
        return f"{self._config['job']}.rollout_proxy" if containerize else self._config["rollout"]["proxy"]["host"]

    def _get_train_proxy_host(self, containerize: bool = False):
        return f"{self._config['job']}.train_proxy" if containerize else self._config["training"]["proxy"]["host"]
