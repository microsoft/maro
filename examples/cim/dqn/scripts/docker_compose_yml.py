# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import yaml
from copy import deepcopy
from os.path import dirname, join, realpath

path = realpath(__file__)
script_dir = dirname(path)
cim_dqn_sync_dir = join(dirname(script_dir), "sync_mode")
cim_dqn_dir = dirname(cim_dqn_sync_dir)
cim_dir = dirname(cim_dqn_dir)
root_dir = dirname(dirname(cim_dir))
maro_rl_dir = join(root_dir, "maro", "rl")
maro_comm_dir = join(root_dir, "maro", "communication")
config_path = join(cim_dqn_dir, "config.yml")
dockerfile_path = join(root_dir, "docker_files", "dev.df")

with open(config_path, "r") as fp:
    config = yaml.safe_load(fp)
    num_trainers = config["policy_manager"]["num_trainers"]
    redis_host = config["redis"]["host"]

docker_compose_manifest = {"version": "3.9", "services": {"redis": {"image": "redis:6", "container_name": redis_host}}}
common_spec = {
    "build": {"context": root_dir, "dockerfile": dockerfile_path},
    "image": "maro-cim",
    "volumes": [
        f"{cim_dir}:/maro/cim",
        f"{maro_rl_dir}:/maro/maro/rl",
        f"{maro_comm_dir}:/maro/maro/communication"
    ]
}

# trainer spec
if config["policy_manager"]["policy_training_mode"] == "multi-node":
    for i in range(num_trainers):
        trainer_id = f"TRAINER.{i}"
        trainer_spec = deepcopy(common_spec)
        del trainer_spec["build"]
        trainer_spec["command"] = "python3 /maro/cim/dqn/policy_manager/trainer.py"
        trainer_spec["container_name"] = trainer_id
        trainer_spec["environment"] = [f"TRAINERID={trainer_id}"]
        docker_compose_manifest["services"][trainer_id] = trainer_spec

# learner_spec
docker_compose_manifest["services"]["learner"] = {
    **common_spec, 
    **{"container_name": "learner", "command": "python3 /maro/cim/dqn/sync_mode/learner.py"}
}

# rollout worker spec
if config["roll_out"]["mode"] == "multi-node":
    for i in range(config["roll_out"]["num_workers"]):
        actor_id = f"ROLLOUT_WORKER.{i}"
        actor_spec = deepcopy(common_spec)
        del actor_spec["build"]
        actor_spec["command"] = "python3 /maro/cim/dqn/sync_mode/rollout_worker.py"
        actor_spec["container_name"] = actor_id
        actor_spec["environment"] = [f"WORKERID={actor_id}"]
        docker_compose_manifest["services"][actor_id] = actor_spec

with open(join(cim_dqn_dir, "docker-compose.yml"), "w") as fp:
    yaml.safe_dump(docker_compose_manifest, fp)
