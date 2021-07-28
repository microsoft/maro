# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import yaml
from copy import deepcopy
from os import makedirs
from os.path import dirname, join, realpath

path = realpath(__file__)
k8s_script_dir = dirname(path)
rl_example_dir = dirname(dirname(k8s_script_dir))
root_dir = dirname(dirname(rl_example_dir))
config_path = join(rl_example_dir, "workflows", "config.yml")
mnt_path = "/rltest"

k8s_manifest_dir = join(k8s_script_dir, "manifests")
makedirs(k8s_manifest_dir, exist_ok=True)

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
                "nodeSelector": {"agentpool": "sc"},
                "restartPolicy": "Never",
                # "imagePullSecrets": [{"name": "emptysecret"}],
                "volumes": [{
                    "name": "maro",
                    "azureFile": {
                        "secretName": "azure-secret",
                        "shareName": "rltest",
                        "readOnly": False
                    }
                }]
            }
        }
    }
}

common_container_spec = {
    "image": "marorl.azurecr.cn/marorl",
    "imagePullPolicy": "Always",
    "volumeMounts": [
        {"name": "maro", "mountPath": mnt_path}
    ]
}

common_env = [
    {"name": "REDISHOST", "value": config["redis"]["host"]},
    {"name": "REDISPORT", "value": redis_port_str},
    {"name": "JOB", "value": config["job"]},
    {"name": "SCENARIO", "value": config["scenario"]},
    {"name": "MODE", "value": config["mode"]},
    {"name": "EXPDIST", "value": "1" if config["rollout_experience_distribution"] else "0"}
]

if config["mode"] == "sync":
    common_env.append({"name": "NUMWORKERS", "value": str(config["sync"]["num_rollout_workers"])})
else:
    common_env.append({"name": "NUMACTORS", "value": str(config["async"]["num_actors"])})

# trainer spec
if config["policy_manager"]["train_mode"] == "multi-node":
    for trainer_id in range(config["policy_manager"]["num_trainers"]):
        job_name = f"trainer-{trainer_id}"
        trainer_manifest = deepcopy(common)
        trainer_manifest["metadata"] = {"name": job_name}
        trainer_manifest["spec"]["template"]["spec"]["containers"] = [
            {   
                **deepcopy(common_container_spec),
                **{
                    "name": job_name,
                    "command": ["python3", join(mnt_path, 'workflows/policy_manager/trainer.py')],
                    "env": [
                        {"name": "TRAINERID", "value": str(trainer_id)},
                        {"name": "TRAINGROUP", "value": config["policy_manager"]["train_group"]}
                    ] + common_env
                }
            }
        ]

        with open(join(k8s_manifest_dir, f"trainer_{trainer_id}.yml"), "w") as fp:
            yaml.safe_dump(trainer_manifest, fp)

mode = config["mode"]
if mode == "sync":
    # learner_spec
    learner_manifest = deepcopy(common)
    learner_manifest["metadata"] = {"name": "learner"}
    learner_manifest["spec"]["template"]["spec"]["containers"] = [
        {   
            **deepcopy(common_container_spec),
            **{
                "name": "learner",
                "command": ["python3", join(mnt_path, 'workflows/synchronous/learner.py')],
                "env": [
                    {"name": "ROLLOUTMODE", "value": config["sync"]["rollout_mode"]},
                    {"name": "NUMSTEPS", "value": str(config["num_steps"])},
                    {"name": "MAXLAG", "value": str(config["max_lag"])},
                    {"name": "MINFINISH", "value": str(config["sync"]["min_finished_workers"])},
                    {"name": "MAXEXRECV", "value": str(config["sync"]["max_extra_recv_tries"])},
                    {"name": "MAXRECVTIMEO", "value": str(config["sync"]["extra_recv_timeout"])},
                    {"name": "ROLLOUTGROUP", "value": config["sync"]["rollout_group"]},
                    {"name": "NUMEPISODES", "value": str(config["num_episodes"])},
                    {"name": "EVALSCH", "value": str(config["eval_schedule"])},
                    {"name": "TRAINMODE", "value": config["policy_manager"]["train_mode"]},
                    {"name": "TRAINGROUP", "value": config["policy_manager"]["train_group"]},
                    {"name": "NUMTRAINERS", "value": str(config["policy_manager"]["num_trainers"])}
                ] + common_env
            }
        }
    ]
    with open(join(k8s_manifest_dir, "learner.yml"), "w") as fp:
        yaml.safe_dump(learner_manifest, fp)

    # rollout worker spec
    if config["sync"]["rollout_mode"] == "multi-node":
        for worker_id in range(config["sync"]["num_rollout_workers"]):
            job_name = f"rollout-worker-{worker_id}"
            worker_manifest = deepcopy(common)
            worker_manifest["metadata"] = {"name": job_name}
            worker_manifest["spec"]["template"]["spec"]["containers"] = [
                {
                    **deepcopy(common_container_spec),
                    **{
                        "name": job_name,
                        "command": ["python3", join(mnt_path, 'workflows/synchronous/rollout_worker.py')],
                        "env": [
                            {"name": "WORKERID", "value": str(worker_id)},
                            {"name": "ROLLOUTGROUP", "value": config["sync"]["rollout_group"]},
                            {"name": "EVALSCH", "value": str(config["eval_schedule"])}
                        ] + common_env
                    }
                }
            ]
            with open(join(k8s_manifest_dir, f"rollout_worker_{worker_id}.yml"), "w") as fp:
                yaml.safe_dump(worker_manifest, fp)
elif mode == "async":
    # policy server spec
    server_manifest = deepcopy(common)
    server_manifest["metadata"]= {"name": "policy-server"}
    server_manifest["spec"]["template"]["spec"]["containers"] = [
        {
            **deepcopy(common_container_spec),
            **{
                "name": "policy-server",
                "command": ["python3", join(mnt_path, 'workflows/asynchronous/policy_server.py')],
                "env": [
                    {"name": "GROUP", "value": config["async"]["group"]},
                    {"name": "MAXLAG", "value": str(config["max_lag"])}
                ] + common_env
            }
        }
    ]
    with open(join(k8s_manifest_dir, "policy_server.yml"), "w") as fp:
        yaml.safe_dump(server_manifest, fp)

    # actor spec
    for actor_id in range(config["async"]["num_actors"]):
        actor_manifest = deepcopy(common)
        actor_manifest["metadata"] = {"name": "actor"}
        actor_manifest["spec"]["template"]["spec"]["containers"] = [
            {
                **deepcopy(common_container_spec),
                **{
                    "name": f"actor.{actor_id}",
                    "command": ["python3", join(mnt_path, 'workflows/asynchronous/actor.py')],
                    "env": [
                        {"name": "ACTORID", "value": str(actor_id)},
                        {"name": "GROUP", "value": config["async"]["group"]},
                        {"name": "NUMEPISODES", "value": str(config["num_episodes"])},
                        {"name": "NUMSTEPS", "value": str(config["num_steps"])},
                        {"name": "EVALSCH", "value": str(config["eval_schedule"])}
                    ] + common_env
                }
            }
        ]
        with open(join(k8s_manifest_dir, f"actor_{actor_id}.yml"), "w") as fp:
            yaml.safe_dump(actor_manifest, fp)
else:
    raise ValueError(f"mode must be 'sync' or 'async', got {mode}")
