# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from os.path import join

from maro.utils.utils import LOCAL_MARO_ROOT

DEFAULT_SCRIPT = {
    "main": "main.py",
    "rollout_proxy": "rollout_proxy.py",
    "rollout_worker": "rollout_worker.py",
    "dispatcher": "dispatcher.py",
    "train_worker": "train_worker.py"
}
MARO_ROOT_IN_CONTAINER = "/maro"


def get_script_path(component: str, containerized: bool = False):
    root = MARO_ROOT_IN_CONTAINER if containerized else LOCAL_MARO_ROOT
    workflow_path = join(root, "maro", "rl", "workflows")
    return join(workflow_path, DEFAULT_SCRIPT[component.split("-")[0]])


def format_env_vars(env: dict, mode: str = "proc"):
    if mode == "proc":
        return env

    if mode == "docker":
        env_opt_list = []
        for key, val in env.items():
            env_opt_list.extend(["--env", f"{key}={val}"])
        return env_opt_list

    if mode == "docker-compose":
        return [f"{key}={val}" for key, val in env.items()]

    if mode == "k8s":
        return [{"name": key, "value": val} for key, val in env.items()]

    raise ValueError(f"'mode' should be one of 'proc', 'docker', 'docker-compose', 'k8s', got {mode}")


def get_mnt_path_in_container(which: str):
    if which not in {"scenario", "loadpoint", "checkpoints", "logs"}:
        raise ValueError("'which' parameter must be one of 'scenario', 'loadpoint', 'checkpoints', 'logs'")

    return f"/{which}"


def get_path_env(config, containerized: bool = False):
    # get path-related environment variables from config
    if containerized:
        env = {"SCENARIO_PATH": get_mnt_path_in_container("scenario")}
        if "load_path" in config:
            env["LOAD_PATH"] = get_mnt_path_in_container("loadpoint")
        if "checkpoint_path" in config:
            env["CHECKPOINT_PATH"] = get_mnt_path_in_container("checkpoints")
        if "log_path" in config:
            env["LOG_PATH"] = get_mnt_path_in_container("logs")
    else:
        env = {"SCENARIO_PATH": config["scenario_path"]}
        if "load_path" in config:
            env["LOAD_PATH"] = config["load_path"]
        if "checkpoint_path" in config:
            env["CHECKPOINT_PATH"] = config["checkpoint_path"]
        if "log_path" in config:
            env["LOG_PATH"] = config["log_path"]

    return env


def get_rl_component_env_vars(config, containerized: bool = False):
    component_env = {
        "main": {
            "JOB": config["job"],
            "MODE": "single",
            "NUM_EPISODES": str(config["num_episodes"]),
            "EVAL_SCHEDULE": str(config["eval_schedule"]),
            "ROLLOUT_MODE": config["rollout"]["mode"],
            "TRAIN_MODE": config["training"]["mode"],
            **get_path_env(config, containerized=containerized)
        }
    }
    if "num_steps" in config:
        component_env["main"]["NUM_STEPS"] = str(config["num_steps"])

    if config["rollout"]["mode"] == "parallel":
        proxy_host = f"{config['job']}.{config['rollout']['proxy']['host']}"
        proxy_frontend_port = str(config["rollout"]["proxy"]["frontend"])
        proxy_backend_port = str(config["rollout"]["proxy"]["backend"])
        num_rollout_workers = config["rollout"]["proxy"]["num_workers"]
        component_env["main"].update({
            "ROLLOUT_PARALLELISM": config["rollout"]["parallelism"],
            "ROLLOUT_PROXY_HOST": proxy_host,
            "ROLLOUT_PROXY_FRONTEND_PORT": proxy_frontend_port
        })

        # optional settings for parallel rollout
        if "eval_parallelism" in config["rollout"]:
            component_env["main"]["EVAL_PARALLELISM"] = config["rollout"]["eval_parallelism"]
        if "min_env_samples" in config["rollout"]:
            component_env["main"]["MIN_ENV_SAMPLES"] = config["rollout"]["min_env_samples"]
        if "grace_factor" in config["rollout"]:
            component_env["main"]["GRACE_FACTOR"] = config["rollout"]["grace_factor"]

        component_env["rollout_proxy"] = {
            "NUM_ROLLOUT_WORKERS": num_rollout_workers,
            "ROLLOUT_PROXY_FRONTEND_PORT": proxy_frontend_port,
            "ROLLOUT_PROXY_BACKEND_PORT": proxy_backend_port
        }
        component_env.update({
            f"rollout_worker-{i}": {
                "ID": str(i), "ROLLOUT_PROXY_HOST": proxy_host, "ROLLOUT_PROXY_BACKEND_PORT": proxy_backend_port,
                **get_path_env(config, containerized=containerized)
            }
            for i in range(num_rollout_workers)
        })

    if config["training"]["mode"] == "parallel":
        dispatcher_host = f"{config['job']}.{config['training']['dispatching']['host']}"
        dispatcher_frontend_port = str(config["training"]["dispatching"]["frontend"])
        dispatcher_backend_port = str(config["training"]["dispatching"]["backend"])
        num_workers = config["training"]["dispatching"]["num_workers"]
        component_env["main"].update({
            "DISPATCHER_HOST": dispatcher_host, "DISPATCHER_FRONTEND_PORT": dispatcher_frontend_port
        })
        component_env["dispatcher"] = {
            "DISPATCHER_FRONTEND_PORT": dispatcher_frontend_port,
            "DISPATCHER_BACKEND_PORT": dispatcher_backend_port
        }
        component_env.update({
            f"train_worker-{i}": {
                "ID": str(i),
                "JOB": config["job"],
                "DISPATCHER_HOST": dispatcher_host,
                "DISPATCHER_BACKEND_PORT": dispatcher_backend_port,
                **get_path_env(config, containerized=containerized)
            }
            for i in range(num_workers)
        })

    return component_env
