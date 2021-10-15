# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import shutil
import yaml
from copy import deepcopy
from os import makedirs
from os.path import dirname, exists, join, realpath


path = realpath(__file__)
k8s_script_dir = dirname(path)
rl_example_dir = dirname(dirname(k8s_script_dir))
root_dir = dirname(dirname(rl_example_dir))
config_path = join(rl_example_dir, "config.yml")

k8s_manifest_dir = join(k8s_script_dir, "manifests")
if exists(k8s_manifest_dir):
    shutil.rmtree(k8s_manifest_dir)

makedirs(k8s_manifest_dir)

with open(config_path, "r") as fp:
    config = yaml.safe_load(fp)

redis_deployment_manifest = {
    "apiVersion": "apps/v1",
    "kind": "Deployment",
    "metadata": {
        "name": config["redis"]["host"],
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
                        "ports": [{"containerPort": config["redis"]["port"]}]
                    }
                ]
            }
        }
    }
}

with open(join(k8s_manifest_dir, f"redis_deployment.yml"), "w") as fp:
    yaml.safe_dump(redis_deployment_manifest, fp)

redis_service_manifest = {
    "apiVersion": "v1",
    "kind": "Service",
    "metadata": {
        "name": config["redis"]["host"],
        "labels": {"app": "redis"}
    },
    "spec": {
        "ports": [{"port": config["redis"]["port"], "targetPort": config["redis"]["port"]}],
        "selector": {"app": "redis"}
    }
}

with open(join(k8s_manifest_dir, f"redis_service.yml"), "w") as fp:
    yaml.safe_dump(redis_service_manifest, fp)

redis_port_str = str(config["redis"]["port"])
common = {
    "apiVersion": "batch/v1",
    "kind": "Job",
    "spec": {
        "template": {
            "spec": {
                "nodeSelector": {"agentpool": config["aks"]["agent_pool"]},
                "restartPolicy": "Never",
                # "imagePullSecrets": [{"name": "emptysecret"}],
                "volumes": [{
                    "name": "maro",
                    "azureFile": {
                        "secretName": "azure-secret",
                        "shareName": config["aks"]["file_share"],
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
    "volumeMounts": [
        {"name": config["aks"]["storage_account"], "mountPath": config["aks"]["file_share"]}
    ]
}

common_envs = [
    {"name": "REDISHOST", "value": config["redis"]["host"]},
    {"name": "REDISPORT", "value": redis_port_str},
    {"name": "JOB", "value": config["job"]},
    {"name": "SCENARIODIR", "value": config["scenario_dir"]},
    {"name": "SCENARIO", "value": config["scenario"]},
    {"name": "MODE", "value": config["mode"]},
    {"name": "POLICYMANAGERTYPE", "value": config["policy_manager"]["type"]}
]

if "load_policy_dir" in config:
    common_envs.append({"name": "LOADDIR", "value": config["load_policy_dir"]})
if "checkpoint_dir" in config:
    common_envs.append({"name": "CHECKPOINTDIR", "value": config["checkpoint_dir"]})
if "log_dir" in config:
    common_envs.append({"name": "LOGDIR", "value": config["log_dir"]})

if config["policy_manager"]["type"] == "distributed":
    common_envs.append({"name": "POLICYGROUP", "value": "-".join([config["job"], "policies"])})
    common_envs.append({"name": "NUMHOSTS", "value": str(config["policy_manager"]["distributed"]["num_hosts"])})

workflow_dir_in_container = "/maro/maro/rl/workflows"
# get script paths
if "scripts" in config and "main" in config["scripts"]:
    main_path = config["scripts"]["main"]
else:
    main_path = join(workflow_dir_in_container, "main.py")

if "scripts" in config and "rollout_worker" in config["scripts"]:
    rollout_worker_path = config["scripts"]["rollout_worker"]
else:
    rollout_worker_path = join(workflow_dir_in_container, "rollout.py")

if "scripts" in config and "policy_host" in config["scripts"]:
    policy_host_path = config["scripts"]["policy_host"]
else:
    policy_host_path = join(workflow_dir_in_container, "policy_host.py")

if "scripts" in config and "policy_server" in config["scripts"]:
    policy_server_path = config["scripts"]["policy_server"]
else:
    policy_server_path = join(workflow_dir_in_container, "policy_manager.py")

if "scripts" in config and "actor" in config["scripts"]:
    actor_path = config["scripts"]["actor"]
else:
    actor_path = join(workflow_dir_in_container, "rollout.py")


# policy host spec
if config["policy_manager"]["type"] == "distributed":
    for host_id in range(config["policy_manager"]["distributed"]["num_hosts"]):
        job_name = f"policy-host-{host_id}"
        host_manifest = deepcopy(common)
        host_manifest["metadata"] = {"name": job_name}
        host_manifest["spec"]["template"]["spec"]["containers"] = [
            {   
                **deepcopy(common_container_spec),
                **{
                    "name": job_name,
                    "command": ["python3", policy_host_path],
                    "env": [{"name": "HOSTID", "value": str(host_id)}] + common_envs
                }
            }
        ]

        with open(join(k8s_manifest_dir, f"policy_host_{host_id}.yml"), "w") as fp:
            yaml.safe_dump(host_manifest, fp)

mode = config["mode"]
if mode == "sync":
    # main process spec
    main_manifest = deepcopy(common)
    main_manifest["metadata"] = {"name": "main"}
    rollout_group = "-".join([config["job"], "rollout"])
    envs = [
        {"name": "ROLLOUTTYPE", "value": config["sync"]["rollout_type"]},
        {"name": "NUMEPISODES", "value": str(config["num_episodes"])},
        {"name": "NUMROLLOUTS", "value": str(config["sync"]["num_rollouts"])}
    ]
    if "num_steps" in config:
        envs.append({"NUMSTEPS": str(config["num_steps"])})
    if "eval_schedule" in config:
        envs.append({"name": "EVALSCH", "value": str(config["eval_schedule"])})
    if config["sync"]["rollout_type"] == "distributed":
        envs.append({"name": "ROLLOUTGROUP", "value": rollout_group})
        if "min_finished_workers" in config["sync"]["distributed"]:
            envs.append({"name": "MINFINISH", "value": str(config["sync"]["distributed"]["min_finished_workers"])})
        if "max_extra_recv_tries" in config["sync"]["distributed"]:
            envs.append({"name": "MAXEXRECV", "value": str(config["sync"]["distributed"]["max_extra_recv_tries"])})
        if "extra_recv_timeout" in config["sync"]["distributed"]:
            envs.append({"name": "MAXRECVTIMEO", "value": str(config["sync"]["distributed"]["extra_recv_timeout"])})

    if "num_eval_rollouts" in config["sync"]:
        envs.append({"name": "NUMEVALROLLOUTS", "value": str(config["sync"]["num_eval_rollouts"])})
    main_manifest["spec"]["template"]["spec"]["containers"] = [
        {   
            **deepcopy(common_container_spec),
            **{
                "name": "main",
                "command": ["python3", main_path],
                "env": envs + common_envs
            }
        }
    ]
    with open(join(k8s_manifest_dir, "main.yml"), "w") as fp:
        yaml.safe_dump(main_manifest, fp)

    # rollout worker spec
    if config["sync"]["rollout_type"] == "distributed":
        for worker_id in range(config["sync"]["num_rollouts"]):
            job_name = f"rollout-worker-{worker_id}"
            worker_manifest = deepcopy(common)
            worker_manifest["metadata"] = {"name": job_name}
            worker_manifest["spec"]["template"]["spec"]["containers"] = [
                {
                    **deepcopy(common_container_spec),
                    **{
                        "name": job_name,
                        "command": ["python3", rollout_worker_path],
                        "env": [
                            {"name": "WORKERID", "value": str(worker_id)},
                            {"name": "ROLLOUTGROUP", "value": rollout_group}
                        ] + common_envs
                    }
                }
            ]
            with open(join(k8s_manifest_dir, f"rollout_worker_{worker_id}.yml"), "w") as fp:
                yaml.safe_dump(worker_manifest, fp)
elif mode == "async":
    # policy server spec
    server_manifest = deepcopy(common)
    server_manifest["metadata"]= {"name": "policy-server"}
    envs = [
        {"name": "GROUP", "value": config["job"]},
        {"name": "NUMROLLOUTS", "value": str(config["async"]["num_rollouts"])},
    ]
    if "max_lag" in config["async"]:
        envs.append({"name": "MAXLAG", "value": str(config["async"]["max_lag"])})
    server_manifest["spec"]["template"]["spec"]["containers"] = [
        {
            **deepcopy(common_container_spec),
            **{
                "name": "policy-server",
                "command": ["python3", policy_server_path],
                "env": envs + common_envs
            }
        }
    ]
    with open(join(k8s_manifest_dir, "policy_server.yml"), "w") as fp:
        yaml.safe_dump(server_manifest, fp)

    # actor spec
    for actor_id in range(config["async"]["num_rollouts"]):
        actor_manifest = deepcopy(common)
        actor_manifest["metadata"] = {"name": "actor"}
        envs = [
            {"name": "ACTORID", "value": str(actor_id)},
            {"name": "GROUP", "value": config["job"]},
            {"name": "NUMEPISODES", "value": str(config["num_episodes"])}
        ]
        if "num_steps" in config:
            envs.append({"name": "NUMSTEPS", "value": str(config["num_steps"])})
        actor_manifest["spec"]["template"]["spec"]["containers"] = [
            {
                **deepcopy(common_container_spec),
                **{
                    "name": f"actor.{actor_id}",
                    "command": ["python3", actor_path],
                    "env": envs + common_envs
                }
            }
        ]
        with open(join(k8s_manifest_dir, f"actor_{actor_id}.yml"), "w") as fp:
            yaml.safe_dump(actor_manifest, fp)
else:
    raise ValueError(f"mode must be 'sync' or 'async', got {mode}")
