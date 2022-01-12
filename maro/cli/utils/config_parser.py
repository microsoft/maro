# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from os.path import join

from maro.utils.utils import LOCAL_MARO_ROOT

DEFAULT_SCRIPT = {
    "main": "main.py",
    # "rolloutworker": "rollout.py",
    # "policyhost": "policy_host.py",
    # "policyserver": "policy_manager.py",
    # "actor": "rollout.py",
    # "task_queue": "task_queue.py",
    # "gradworker": "grad_worker.py"
    "train_worker": "train_worker.py",
    "dispatcher": "dispatcher.py"
}
MARO_ROOT_IN_CONTAINER = "/maro"


def get_script_path(component: str, containerized: bool = False):
    root = MARO_ROOT_IN_CONTAINER if containerized else LOCAL_MARO_ROOT
    workflow_path = join(root, "maro", "rl_v3", "workflows")
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
        raise ValueError(f"'which' parameter must be one of 'scenario', 'loadpoint', 'checkpoints', 'logs'")

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
            "ROLLOUT_MODE": config["rollout_mode"],
            "TRAIN_MODE": config["train_mode"],
            **get_path_env(config, containerized=containerized)
        }
    }
    if "num_steps" in config:
        component_env["main"]["NUM_STEPS"] = str(config["num_steps"])

    if config["train_mode"] == "parallel":
        dispatcher_host = f"{config['job']}.{config['distributed']['dispatcher_host']}"
        dispatcher_frontend_port = str(config["distributed"]["dispatcher_frontend_port"])
        dispatcher_backend_port = str(config["distributed"]["dispatcher_backend_port"])
        num_workers = config["distributed"]["num_train_workers"]
        component_env["main"].update({
            "DISPATCHER_HOST": dispatcher_host, "DISPATCHER_FRONTEND_PORT": dispatcher_frontend_port
        })
        component_env["dispatcher"] = {
            "NUM_WORKERS": str(num_workers),
            "DISPATCHER_FRONTEND_PORT": dispatcher_frontend_port,
            "DISPATCHER_BACKEND_PORT": dispatcher_backend_port
        }
        component_env.update({
            f"train_worker-{i}": {
                "ID": str(i), "DISPATCHER_HOST": dispatcher_host, "DISPATCHER_BACKEND_PORT": dispatcher_backend_port,
                **get_path_env(config, containerized=containerized)
            }
            for i in range(num_workers)
        })

    return component_env

    # if "DATA_PARALLELISM" in common and int(common["DATA_PARALLELISM"]) > 1:
    #     component_env["task_queue"] = common
    #     component_env.update({
    #         f"gradworker-{worker_id}": {**common, "WORKER_ID": str(worker_id)}
    #         for worker_id in range(int(common["DATA_PARALLELISM"]))
    #     })

    # if config["mode"] == "sync":
    #     env = {
    #         "ROLLOUT_TYPE": config["sync"]["rollout_type"],
    #         "NUM_EPISODES": str(config["num_episodes"]),
    #         "NUM_ROLLOUTS": str(config["sync"]["num_rollouts"])
    #     }

    #     if "num_steps" in config:
    #         env["NUM_STEPS"] = str(config["num_steps"])
    #     if "eval_schedule" in config:
    #         env["EVAL_SCHEDULE"] = str(config["eval_schedule"])
    #     if config["sync"]["rollout_type"] == "distributed":
    #         env["ROLLOUT_GROUP"] = "-".join([config["job"], "rollout"])
    #         if "min_finished_workers" in config["sync"]["distributed"]:
    #             env["MIN_FINISHED_WORKERS"] = str(config["sync"]["distributed"]["min_finished_workers"])
    #         if "max_extra_recv_tries" in config["sync"]["distributed"]:
    #             env["MAX_EXTRA_RECV_TRIES"] = str(config["sync"]["distributed"]["max_extra_recv_tries"])
    #         if "extra_recv_timeout" in config["sync"]["distributed"]:
    #             env["MAX_RECV_TIMEO"] = str(config["sync"]["distributed"]["extra_recv_timeout"])
    #         component_env.update({
    #             f"rolloutworker-{worker_id}":
    #                 {**common, "WORKER_ID": str(worker_id), "ROLLOUT_GROUP": "-".join([config["job"], "rollout"])}
    #             for worker_id in range(config["sync"]["num_rollouts"])
    #         })

    #     if "num_eval_rollouts" in config["sync"]:
    #         env["NUM_EVAL_ROLLOUTS"] = str(config["sync"]["num_eval_rollouts"])

    #     component_env["main"] = {**common, **env}
    #     return component_env

    # if config["mode"] == "async":
    #     for actor_id in range(config["async"]["num_rollouts"]):
    #         actor_env = {
    #             "ACTOR_ID": str(actor_id),
    #             "GROUP": config["job"],
    #             "NUM_EPISODES": str(config["num_episodes"])
    #         }
    #         if "num_steps" in config:
    #             actor_env["NUM_STEPS"] = config["num_steps"]
    #         component_env[f"actor-{actor_id}"] = {**common, **actor_env}

    #     server_env = {"GROUP": config["job"], "NUM_ROLLOUTS": str(config["async"]["num_rollouts"])}
    #     if "max_lag" in config["async"]:
    #         server_env["MAX_LAG"] = str(config["async"]["max_lag"])
    #     component_env["policyserver"] = {**common, **server_env}

    #     return component_env

    # raise ValueError(f"'mode' must be 'single', 'sync' or 'async', got {config['mode']}")
