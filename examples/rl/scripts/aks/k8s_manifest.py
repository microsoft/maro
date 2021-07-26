# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import yaml
from copy import deepcopy
from os.path import dirname, join, realpath

path = realpath(__file__)
k8s_script_dir = dirname(path)
rl_example_dir = dirname(dirname(k8s_script_dir))
root_dir = dirname(dirname(rl_example_dir))
config_path = join(rl_example_dir, "workflows", "config.yml")
mnt_path = "/maro/rl_examples"

with open(config_path, "r") as fp:
    config = yaml.safe_load(fp)
    redis_host = config["redis"]["host"]

common = {
    "apiVersion": "batch/v1",
    "kind": "Job",
    "spec": {
        "template": {
            "spec": {
                "nodeSelector": {"agentpool": "marosc"},
                "restartPolicy": "Never",
                "volumes": [{
                    "name": "maro",
                    "azureFile": {
                        "secretName": "azure-secret",
                        "shareName": "rltest",
                        "readOnly": "false"
                    }
                }]
            }
        }
    }
}

common_container_spec = {
    "image": "maroacr.azurecr.cn/maro-sc",
    "imagePullPolicy": "Always",
    "volumeMounts": [
        {"name": "maro-sc", "mountPath": mnt_path}
    ]
}

# trainer spec
if config["policy_manager"]["train_mode"] == "multi-node":
    trainer_manifest = deepcopy(common)
    trainer_manifest["spec"]["template"]["spec"]["containers"] = [
        {   
            **deepcopy(common_container_spec),
            **{
                "name": f"trainer.{trainer_id}",
                "command": f"python3 {join(mnt_path, 'workflows/policy_manager/trainer.py')}",
                "env": [{"name": "TRAINERID", "value": trainer_id}]
            }
        } for trainer_id in range(config["policy_manager"]["num_trainers"])
    ]

    with open(join(k8s_script_dir, "trainer.yml"), "w") as fp:
        yaml.safe_dump(trainer_manifest, fp)

mode = config["mode"]
if mode == "sync":
    # learner_spec
    learner_manifest = deepcopy(common)
    learner_manifest["spec"]["template"]["spec"]["containers"] = [
        {   
            **deepcopy(common_container_spec),
            **{
                "name": "learner",
                "command": f"python3 {join(mnt_path, 'workflows/synchronous/learner.py')}",
            }
        }
    ]
    with open(join(k8s_script_dir, "learner.yml"), "w") as fp:
        yaml.safe_dump(learner_manifest, fp)

    # rollout worker spec
    if config["sync"]["rollout_mode"] == "multi-node":
        worker_manifest = deepcopy(common)
        worker_manifest["spec"]["template"]["spec"]["containers"] = [
            {
                **deepcopy(common_container_spec),
                **{
                    "name": f"rollout_worker.{worker_id}",
                    "command": f"python3 {join(mnt_path, 'workflows/synchronous/rollout_worker.py')}",
                    "env": [{"name": "WORKERID", "value": worker_id}]
                }
            } for worker_id in range(config["sync"]["num_rollout_workers"])
        ]
        with open(join(k8s_script_dir, "rollout_worker.yml"), "w") as fp:
            yaml.safe_dump(worker_manifest, fp)
elif mode == "async":
    # policy server spec
    server_manifest = deepcopy(common)
    server_manifest["spec"]["template"]["spec"]["containers"] = [
        {
            **deepcopy(common_container_spec),
            **{
                "name": "policy_server",
                "command": f"python3 {join(mnt_path, 'workflows/asynchronous/policy_server.py')}"
            }
        }
    ]
    with open(join(k8s_script_dir, "policy_server.yml"), "w") as fp:
        yaml.safe_dump(server_manifest, fp)

    # actor spec
    actor_manifest = deepcopy(common)
    actor_manifest["spec"]["template"]["spec"]["containers"] = [
        {
            **deepcopy(common_container_spec),
            **{
                "name": f"actor.{actor_id}",
                "command": f"python3 {join(mnt_path, 'workflows/asynchronous/actor.py')}",
                "env": [{"name": "ACTORID", "value": actor_id}]
            }
        } for actor_id in range(config["async"]["num_actors"])
    ]
    with open(join(k8s_script_dir, "actor.yml"), "w") as fp:
        yaml.safe_dump(actor_manifest, fp)
else: 
    raise ValueError(f"mode must be 'sync' or 'async', got {mode}")
