# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
from os.path import basename, join

from maro.utils.utils import LOCAL_MARO_ROOT


DEFAULT_SCRIPT = {
    "main": "main.py",
    "rolloutworker": "rollout.py",
    "policyhost": "policy_host.py",
    "policyserver": "policy_manager.py",
    "actor": "rollout.py",
    "task_queue": "task_queue.py",
    "gradworker": "grad_worker.py"
}
MARO_ROOT_IN_CONTAINER = "/maro"

# local paths
def get_script_path(component: str, containerized: bool = False):
    root = MARO_ROOT_IN_CONTAINER if containerized else LOCAL_MARO_ROOT
    workflow_dir = join(root, "maro", "rl", "workflows")
    return join(workflow_dir, DEFAULT_SCRIPT[component.split("-")[0]]) 


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


def get_volume_mapping(config):
    mapping = {config["scenario_dir"]: f"/{basename(config['scenario_dir'])}"}
    if "load_policy_dir" in config:
        mapping[config["load_policy_dir"]] = "/load"
    if "checkpoint_dir" in config:
        mapping[config["checkpoint_dir"]] = "/checkpoints"
    if "log_dir" in config:
        mapping[config["log_dir"]] = "/logs"

    return mapping


def get_dir_env(config, containerized: bool = False):
    # get path-related environment variables from config
    if containerized:
        mapping = get_volume_mapping(config)
        env = {"SCENARIODIR": mapping[config["scenario_dir"]]}
        if "load_policy_dir" in config:
            env["LOADDIR"] = mapping[config["load_policy_dir"]]
        if "checkpoint_dir" in config:
            env["CHECKPOINTDIR"] = mapping[config["checkpoint_dir"]]
        if "log_dir" in config:
            env["LOGDIR"] = mapping[config["log_dir"]]
    else:
        env = {"SCENARIODIR": config["scenario_dir"]}
        if "load_policy_dir" in config:
            env["LOADDIR"] = config["load_policy_dir"]
        if "checkpoint_dir" in config:
            env["CHECKPOINTDIR"] = config["checkpoint_dir"]
        if "log_dir" in config:
            env["LOGDIR"] = config["log_dir"]

    return env


def get_common_env_vars(config, containerized: bool = False):
    env = {
        "JOB": config["job"],
        "MODE": config['mode'],
        "POLICYMANAGERTYPE": config["policy_manager"]["type"],
        **get_dir_env(config, containerized=containerized)
    }

    if "data_parallelism" in config:
        env["DATAPARALLELISM"] = str(config["data_parallelism"])
    if config["policy_manager"]["type"] == "distributed":
        env["POLICYGROUP"] = "-".join([config["job"], "policies"])
        env["NUMHOSTS"] = str(config["policy_manager"]["distributed"]["num_hosts"])

    return env


def get_rl_component_env_vars(config, containerized: bool = False):
    if config["mode"] == "single":
        env = {
            "JOB": config["job"],
            "MODE": "single",
            "NUMEPISODES": str(config["num_episodes"]),
            "EVALSCH": str(config["eval_schedule"]),
            **get_dir_env(config, containerized=containerized)
        }
        if "num_steps" in config:
            env["NUMSTEPS"] = str(config["num_steps"])

        return {"main": env}

    component_env = {}
    common = get_common_env_vars(config, containerized=containerized)
    if "POLICYGROUP" in common:
        component_env.update({
            f"policyhost-{host_id}": {**common, "HOSTID": str(host_id)}
            for host_id in range(config["policy_manager"]["distributed"]["num_hosts"])
        })

    if "DATAPARALLELISM" in common and int(common["DATAPARALLELISM"]) > 1:
        component_env["task_queue"] = common
        component_env.update({
            f"gradworker-{worker_id}": {**common, "WORKERID": str(worker_id)}
            for worker_id in range(int(common["DATAPARALLELISM"]))
        })

    if config["mode"] == "sync":
        env = {
            "ROLLOUTTYPE": config["sync"]["rollout_type"],
            "NUMEPISODES": str(config["num_episodes"]),
            "NUMROLLOUTS": str(config["sync"]["num_rollouts"])
        }

        if "num_steps" in config:
            env["NUMSTEPS"] = str(config["num_steps"])
        if "eval_schedule" in config:
            env["EVALSCH"] = str(config["eval_schedule"])
        if config["sync"]["rollout_type"] == "distributed":
            env["ROLLOUTGROUP"] = "-".join([config["job"], "rollout"])
            if "min_finished_workers" in config["sync"]["distributed"]:
                env["MINFINISH"] = str(config["sync"]["distributed"]["min_finished_workers"])
            if "max_extra_recv_tries" in config["sync"]["distributed"]:
                env["MAXEXRECV"] = str(config["sync"]["distributed"]["max_extra_recv_tries"])
            if "extra_recv_timeout" in config["sync"]["distributed"]:
                env["MAXRECVTIMEO"] = str(config["sync"]["distributed"]["extra_recv_timeout"])

        if "num_eval_rollouts" in config["sync"]:
            env["NUMEVALROLLOUTS"] = str(config["sync"]["num_eval_rollouts"])

        component_env["main"] = {**common, **env}
        component_env.update({
            f"rolloutworker-{worker_id}":
                {**common, "WORKERID" : str(worker_id), "ROLLOUTGROUP": "-".join([config["job"], "rollout"])}
            for worker_id in range(config["sync"]["num_rollouts"])
        })

        return component_env

    if config["mode"] == "async":
        for actor_id in range(config["async"]["num_rollouts"]):
            actor_env = {
                "ACTORID": str(actor_id),
                "GROUP": config["job"],
                "NUMEPISODES": str(config["num_episodes"])
            }
            if "num_steps" in config:
                actor_env["NUMSTEPS"] = config["num_steps"]
            component_env[f"actor-{actor_id}"] = {**common, **actor_env}

        server_env = {"GROUP": config["job"], "NUMROLLOUTS": str(config["async"]["num_rollouts"])}
        if "max_lag" in config["async"]:
            server_env["MAXLAG"] = str(config["async"]["max_lag"])
        component_env["policyserver"] = {**common, **server_env}

        return component_env

    raise ValueError(f"'mode' must be 'single', 'sync' or 'async', got {config['mode']}")


def get_k8s_ymls(config):
    redis_host, redis_port = f"{config['job']}-{config['redis']['host']}", config["redis"]["port"]
    component_manifest = {
        "redis_deployment": {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": redis_host,
                "labels": {"app": "redis"}
            },
            "spec": {
                "selector": {
                    "matchLabels": {"app": "redis"}
                },
                "replicas": 1,
                "template": {
                    "metadata": {
                        "labels": {"app": "redis"}
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": "master",
                                "image": "redis:6",
                                "ports": [{"containerPort": redis_port}]
                            }
                        ]
                    }
                }
            }
        },
        "redis_service": {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": redis_host,
                "labels": {"app": "redis"}
            },
            "spec": {
                "ports": [{"port": redis_port, "targetPort": redis_port}],
                "selector": {"app": "redis"}
            }
        }
    }

    volume, file_share = config["aks"]["volume"], config["aks"]["file_share"]
    job_manifest_common = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "spec": {
            "template": {
                "spec": {
                    "nodeSelector": {"agentpool": config["aks"]["agent_pool"]},
                    "restartPolicy": "Never",
                    # "imagePullSecrets": [{"name": "emptysecret"}],
                    "volumes": [{
                        "name": volume,
                        "azureFile": {
                            "secretName": "azure-secret",
                            "shareName": file_share,
                            "readOnly": False
                        }
                    }]
                }
            }
        }
    }

    common_container_spec = {
        "image": config["aks"]["image"],
        "imagePullPolicy": "Always",
        "volumeMounts": [{"name": volume, "mountPath": file_share}]
    }

    for component, env in get_rl_component_env_vars(config).items():
        component_manifest[component] = deepcopy(job_manifest_common)
        component_manifest[component]["metadata"] = {"name": component}
        component_manifest[component]["spec"]["template"]["spec"]["containers"] = [
            {
                **common_container_spec,
                **{
                    "name": component,
                    "command": ["python3", get_script_path(component, containerized=True)],
                    "env": format_env_vars({**env, "REDISHOST": redis_host, "REDISPORT": str(redis_port)}, mode="k8s")
                }
            }
        ]

    return component_manifest
