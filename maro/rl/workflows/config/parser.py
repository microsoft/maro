# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ipaddress
import os
from typing import Dict, Tuple, Union

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

    def _validate(self) -> None:
        if "job" not in self._config:
            raise KeyError(f"{self._validation_err_pfx}: missing field 'job'")
        if "scenario_path" not in self._config:
            raise KeyError(f"{self._validation_err_pfx}: missing field 'scenario_path'")
        if "log_path" not in self._config:
            raise KeyError(f"{self._validation_err_pfx}: missing field 'log_path'")

        self._validate_main_section()
        self._validate_rollout_section()
        self._validate_training_section()

    def _validate_main_section(self) -> None:
        if "main" not in self._config:
            raise KeyError(f"{self._validation_err_pfx}: missing field 'main'")

        if "num_episodes" not in self._config["main"]:
            raise KeyError(f"{self._validation_err_pfx}: missing field 'num_episodes' under section 'main'")

        num_episodes = self._config["main"]["num_episodes"]
        if not isinstance(num_episodes, int) or num_episodes < 1:
            raise ValueError(f"{self._validation_err_pfx}: 'main.num_episodes' must be a positive int")

        num_steps = self._config["main"].get("num_steps", None)
        if num_steps is not None:
            if not isinstance(num_steps, int) or num_steps <= 0:
                raise ValueError(f"{self._validation_err_pfx}: 'main.num_steps' must be a positive int")

        eval_schedule = self._config["main"].get("eval_schedule", None)
        if eval_schedule is not None:
            if (
                not isinstance(eval_schedule, (int, list))
                or isinstance(eval_schedule, int)
                and eval_schedule < 1
                or isinstance(eval_schedule, list)
                and any(not isinstance(val, int) or val < 1 for val in eval_schedule)
            ):
                raise ValueError(
                    f"{self._validation_err_pfx}: 'main.eval_schedule' must be a single positive int or a list of "
                    f"positive ints",
                )

        if "logging" in self._config["main"]:
            self._validate_logging_section("main", self._config["main"]["logging"])

    def _validate_rollout_section(self) -> None:
        if "rollout" not in self._config or not isinstance(self._config["rollout"], dict):
            raise KeyError(f"{self._validation_err_pfx}: missing section 'rollout'")

        # validate parallel rollout config
        if "parallelism" in self._config["rollout"]:
            conf = self._config["rollout"]["parallelism"]
            if "sampling" not in conf:
                raise KeyError(
                    f"{self._validation_err_pfx}: missing field 'sampling' under section 'rollout.parallelism'",
                )

            train_prl = conf["sampling"]
            eval_prl = 1 if "eval" not in conf or conf["eval"] is None else conf["eval"]
            if not isinstance(train_prl, int) or train_prl <= 0:
                raise TypeError(f"{self._validation_err_pfx}: 'rollout.parallelism.sampling' must be a positive int")
            if not isinstance(eval_prl, int) or eval_prl <= 0:
                raise TypeError(f"{self._validation_err_pfx}: 'rollout.parallelism.eval' must be a positive int")
            if max(train_prl, eval_prl) > 1:
                if "controller" not in conf:
                    raise KeyError(
                        f"{self._validation_err_pfx}: missing field 'controller' under section 'rollout.parallelism'",
                    )
                self._validate_rollout_controller_section(conf["controller"])

                # validate optional fields: min_env_samples, grace_factor
                min_env_samples = conf.get("min_env_samples", None)
                if min_env_samples is not None:
                    if not isinstance(min_env_samples, int) or min_env_samples > train_prl:
                        raise ValueError(
                            f"{self._validation_err_pfx}: 'rollout.parallelism.min_env_samples' must be an integer "
                            f"that does not exceed the value of 'rollout.parallelism.sampling': {train_prl}",
                        )

                grace_factor = conf.get("grace_factor", None)
                if grace_factor is not None and not isinstance(grace_factor, (int, float)):
                    raise ValueError(
                        f"{self._validation_err_pfx}: 'rollout.parallelism.grace_factor' must be an int or float",
                    )

                if "logging" in self._config["rollout"]:
                    self._validate_logging_section("rollout", self._config["rollout"]["logging"])

    def _validate_rollout_controller_section(self, conf: dict) -> None:
        if "host" not in conf:
            raise KeyError(
                f"{self._validation_err_pfx}: missing field 'host' under section 'rollout.parallelism.controller'",
            )
        if not isinstance(conf["host"], str):
            raise TypeError(f"{self._validation_err_pfx}: 'rollout.parallelism.controller.host' must be a string")

        # Check that the host string is a valid IP address
        try:
            ipaddress.ip_address(conf["host"])
        except ValueError:
            raise ValueError(
                f"{self._validation_err_pfx}: 'rollout.parallelism.controller.host' is not a valid IP address",
            )

        if "port" not in conf:
            raise KeyError(
                f"{self._validation_err_pfx}: missing field 'port' under section 'rollout.parallelism.controller'",
            )
        if not isinstance(conf["port"], int):
            raise TypeError(f"{self._validation_err_pfx}: 'rollout.parallelism.controller.port' must be an int")

    def _validate_training_section(self) -> None:
        if "training" not in self._config or not isinstance(self._config["training"], dict):
            raise KeyError(f"{self._validation_err_pfx}: missing field 'training'")
        if "mode" not in self._config["training"]:
            raise KeyError(f"{self._validation_err_pfx}: missing field 'mode' under section 'training'")
        if self._config["training"]["mode"] not in {"simple", "parallel"}:
            raise ValueError(
                f"'mode' value under section 'training' must be 'simple' or 'parallel', got {self._config['mode']}",
            )

        if self._config["training"]["mode"] == "parallel":
            if "num_workers" not in self._config["training"]:
                raise KeyError(f"{self._validation_err_pfx}: missing field 'num_workers' under section 'training'")
            if "proxy" not in self._config["training"]:
                raise KeyError(f"{self._validation_err_pfx}: missing field 'proxy' under section 'training'")
            self._validate_train_proxy_section(self._config["training"]["proxy"])
            if "logging" in self._config["training"]:
                self._validate_logging_section("training", self._config["training"]["logging"])

        load_path = self._config["training"].get("load_path", None)
        if load_path is not None and not isinstance(load_path, str):
            raise TypeError(f"{self._validation_err_pfx}: 'training.load_path' must be a string")
        load_episode = self._config["training"].get("load_episode", None)
        if load_episode is not None and not isinstance(load_episode, int):
            raise TypeError(f"{self._validation_err_pfx}: 'training.load_episode' must be a integer")

        if "checkpointing" in self._config["training"]:
            self._validate_checkpointing_section(self._config["training"]["checkpointing"])

    def _validate_train_proxy_section(self, proxy_section: dict) -> None:
        if "host" not in proxy_section:
            raise KeyError(f"{self._validation_err_pfx}: missing field 'host' under section 'proxy'")
        if not isinstance(proxy_section["host"], str):
            raise TypeError(f"{self._validation_err_pfx}: 'training.proxy.host' must be a string")
        # Check that the host string is a valid IP address
        try:
            ipaddress.ip_address(proxy_section["host"])
        except ValueError:
            raise ValueError(f"{self._validation_err_pfx}: 'training.proxy.host' is not a valid IP address")

        if "frontend" not in proxy_section:
            raise KeyError(f"{self._validation_err_pfx}: missing field 'frontend' under section 'proxy'")
        if not isinstance(proxy_section["frontend"], int):
            raise TypeError(f"{self._validation_err_pfx}: 'training.proxy.frontend' must be an int")

        if "backend" not in proxy_section:
            raise KeyError(f"{self._validation_err_pfx}: missing field 'backend' under section 'proxy'")
        if not isinstance(proxy_section["backend"], int):
            raise TypeError(f"{self._validation_err_pfx}: 'training.proxy.backend' must be an int")

    def _validate_checkpointing_section(self, section: dict) -> None:
        if "path" not in section:
            raise KeyError(f"{self._validation_err_pfx}: missing field 'path' under section 'checkpointing'")
        if not isinstance(section["path"], str):
            raise TypeError(f"{self._validation_err_pfx}: 'training.checkpointing.path' must be a string")

        if "interval" in section:
            if not isinstance(section["interval"], int):
                raise TypeError(
                    f"{self._validation_err_pfx}: 'training.checkpointing.interval' must be an int",
                )

    def _validate_logging_section(self, component: str, level_dict: dict) -> None:
        if any(key not in {"stdout", "file"} for key in level_dict):
            raise KeyError(
                f"{self._validation_err_pfx}: fields under section '{component}.logging' must be 'stdout' or 'file'",
            )
        valid_log_levels = set(LEVEL_MAP.keys())
        for key, val in level_dict.items():
            if val not in valid_log_levels:
                raise ValueError(
                    f"{self._validation_err_pfx}: '{component}.logging.{key}' must be one of {valid_log_levels}.",
                )

    def get_path_mapping(self, containerize: bool = False) -> dict:
        """Generate path mappings for a local or containerized environment.

        Args:
            containerize (bool): If true, the paths you specify in the configuration file (which should always be local)
                are mapped to paths inside the containers as follows:
                    local/scenario/path -> "/scenario"
                    local/load/path -> "/loadpoint"
                    local/checkpoint/path -> "/checkpoints"
                    local/log/path -> "/logs"
                Defaults to False.
        """
        log_dir = os.path.dirname(self._config["log_path"])
        path_map = {
            self._config["scenario_path"]: "/scenario" if containerize else self._config["scenario_path"],
            log_dir: "/logs" if containerize else log_dir,
        }

        load_path = self._config["training"].get("load_path", None)
        if load_path is not None:
            path_map[load_path] = "/loadpoint" if containerize else load_path
        if "checkpointing" in self._config["training"]:
            ckpt_path = self._config["training"]["checkpointing"]["path"]
            path_map[ckpt_path] = "/checkpoints" if containerize else ckpt_path

        return path_map

    def get_job_spec(self, containerize: bool = False) -> Dict[str, Tuple[str, Dict[str, str]]]:
        """Generate environment variables for the workflow scripts.

        A doubly-nested dictionary is returned that contains the environment variables for each distributed component.

        Args:
            containerize (bool): If true, the generated environment variables are to be used in a containerized
                environment. Only path-related environment variables are affected by this flag. See the docstring
                for ``get_path_mappings`` for details. Defaults to False.
        """
        path_mapping = self.get_path_mapping(containerize=containerize)
        scenario_path = path_mapping[self._config["scenario_path"]]
        num_episodes = self._config["main"]["num_episodes"]
        main_proc = f"{self._config['job']}.main"
        min_n_sample = self._config["main"].get("min_n_sample", 1)
        env: dict = {
            main_proc: (
                os.path.join(self._get_workflow_path(containerize=containerize), "main.py"),
                {
                    "JOB": self._config["job"],
                    "NUM_EPISODES": str(num_episodes),
                    "MIN_N_SAMPLE": str(min_n_sample),
                    "TRAIN_MODE": self._config["training"]["mode"],
                    "SCENARIO_PATH": scenario_path,
                },
            ),
        }

        main_proc_env = env[main_proc][1]
        if "eval_schedule" in self._config["main"]:
            # If it is an int, it is treated as the number of episodes between two adjacent evaluations. For example,
            # if the total number of episodes is 20 and this is 5, an evaluation schedule of [5, 10, 15, 20]
            # (start from 1) will be generated for the environment variable (as a string). If it is a list, the sorted
            # version of the list will be generated for the environment variable (as a string).
            sch = self._config["main"]["eval_schedule"]
            if isinstance(sch, int):
                main_proc_env["EVAL_SCHEDULE"] = " ".join([str(sch * i) for i in range(1, num_episodes // sch + 1)])
            else:
                main_proc_env["EVAL_SCHEDULE"] = " ".join([str(val) for val in sorted(sch)])

        load_path = self._config["training"].get("load_path", None)
        if load_path is not None:
            env["main"]["LOAD_PATH"] = path_mapping[load_path]
        load_episode = self._config["training"].get("load_episode", None)
        if load_episode is not None:
            env["main"]["LOAD_EPISODE"] = str(load_episode)

        if "checkpointing" in self._config["training"]:
            conf = self._config["training"]["checkpointing"]
            main_proc_env["CHECKPOINT_PATH"] = path_mapping[conf["path"]]
            if "interval" in conf:
                main_proc_env["CHECKPOINT_INTERVAL"] = str(conf["interval"])

        num_steps = self._config["main"].get("num_steps", None)
        if num_steps is not None:
            main_proc_env["NUM_STEPS"] = str(num_steps)

        if "logging" in self._config["main"]:
            main_proc_env.update(
                {
                    "LOG_LEVEL_STDOUT": self.config["main"]["logging"]["stdout"],
                    "LOG_LEVEL_FILE": self.config["main"]["logging"]["file"],
                },
            )

        if "parallelism" in self._config["rollout"]:
            conf = self._config["rollout"]["parallelism"]
            env_sampling_parallelism = conf["sampling"]
            env_eval_parallelism = 1 if "eval" not in conf or conf["eval"] is None else conf["eval"]
        else:
            env_sampling_parallelism = env_eval_parallelism = 1
        rollout_parallelism = max(env_sampling_parallelism, env_eval_parallelism)
        if rollout_parallelism > 1:
            conf = self._config["rollout"]["parallelism"]
            rollout_controller_port = str(conf["controller"]["port"])
            main_proc_env["ENV_SAMPLE_PARALLELISM"] = str(env_sampling_parallelism)
            main_proc_env["ENV_EVAL_PARALLELISM"] = str(env_eval_parallelism)
            main_proc_env["ROLLOUT_CONTROLLER_PORT"] = rollout_controller_port
            # optional settings for parallel rollout
            if "min_env_samples" in self._config["rollout"]:
                main_proc_env["MIN_ENV_SAMPLES"] = str(conf["min_env_samples"])
            if "grace_factor" in self._config["rollout"]:
                main_proc_env["GRACE_FACTOR"] = str(conf["grace_factor"])

            for i in range(rollout_parallelism):
                worker_id = f"{self._config['job']}.rollout_worker-{i}"
                env[worker_id] = (
                    os.path.join(self._get_workflow_path(containerize=containerize), "rollout_worker.py"),
                    {
                        "ID": str(i),
                        "ROLLOUT_CONTROLLER_HOST": self._get_rollout_controller_host(containerize=containerize),
                        "ROLLOUT_CONTROLLER_PORT": rollout_controller_port,
                        "SCENARIO_PATH": scenario_path,
                    },
                )
                if "logging" in self._config["rollout"]:
                    env[worker_id][1].update(
                        {
                            "LOG_LEVEL_STDOUT": self.config["rollout"]["logging"]["stdout"],
                            "LOG_LEVEL_FILE": self.config["rollout"]["logging"]["file"],
                        },
                    )

        if self._config["training"]["mode"] == "parallel":
            conf = self._config["training"]["proxy"]
            producer_host = self._get_train_proxy_host(containerize=containerize)
            proxy_frontend_port = str(conf["frontend"])
            proxy_backend_port = str(conf["backend"])
            num_workers = self._config["training"]["num_workers"]
            env[main_proc][1].update(
                {
                    "TRAIN_PROXY_HOST": producer_host,
                    "TRAIN_PROXY_FRONTEND_PORT": proxy_frontend_port,
                },
            )
            env[f"{self._config['job']}.train_proxy"] = (
                os.path.join(self._get_workflow_path(containerize=containerize), "train_proxy.py"),
                {"TRAIN_PROXY_FRONTEND_PORT": proxy_frontend_port, "TRAIN_PROXY_BACKEND_PORT": proxy_backend_port},
            )
            for i in range(num_workers):
                worker_id = f"{self._config['job']}.train_worker-{i}"
                env[worker_id] = (
                    os.path.join(self._get_workflow_path(containerize=containerize), "train_worker.py"),
                    {
                        "ID": str(i),
                        "TRAIN_PROXY_HOST": producer_host,
                        "TRAIN_PROXY_BACKEND_PORT": proxy_backend_port,
                        "SCENARIO_PATH": scenario_path,
                    },
                )
                if "logging" in self._config["training"]:
                    env[worker_id][1].update(
                        {
                            "LOG_LEVEL_STDOUT": self.config["training"]["logging"]["stdout"],
                            "LOG_LEVEL_FILE": self.config["training"]["logging"]["file"],
                        },
                    )

        # All components write logs to the same file
        log_dir, log_file = os.path.split(self._config["log_path"])
        for _, vars in env.values():
            vars["LOG_PATH"] = os.path.join(path_mapping[log_dir], log_file)

        return env

    def _get_workflow_path(self, containerize: bool = False) -> str:
        if containerize:
            return "/maro/maro/rl/workflows"
        else:
            return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    def _get_rollout_controller_host(self, containerize: bool = False) -> str:
        if containerize:
            return f"{self._config['job']}.main"
        else:
            return self._config["rollout"]["parallelism"]["controller"]["host"]

    def _get_train_proxy_host(self, containerize: bool = False) -> str:
        return f"{self._config['job']}.train_proxy" if containerize else self._config["training"]["proxy"]["host"]
