# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import yaml
from copy import deepcopy
from os.path import dirname, join, realpath


path = realpath(__file__)
script_dir = dirname(path)
cim_dqn_async_dir = dirname(script_dir)
cim_dqn_dir = dirname(cim_dqn_async_dir)
cim_dir = dirname(cim_dqn_dir)
root_dir = dirname(dirname(cim_dir))
maro_rl_dir = join(root_dir, "maro", "rl")
maro_comm_dir = join(root_dir, "maro", "communication")
config_path = join(cim_dqn_dir, "config.yml")
dockerfile_path = join(root_dir, "docker_files", "dev.df")

with open(config_path, "r") as fp:
    config = yaml.safe_load(fp)
    num_actors = config["async"]["num_actors"]
    num_trainers = config["policy_manager"]["num_trainers"]
    redis_host = config["redis"]["host"]

docker_compose_manifest = {
    "version": "3.9",
    "services": {
        "redis": {"image": "redis:6", "container_name": redis_host},
        "policy_server": {
            "build": {"context": root_dir, "dockerfile": dockerfile_path},
            "image": "maro-cim",
            "container_name": "policy_server",
            "volumes": [
                f"{cim_dir}:/maro/cim",
                f"{maro_rl_dir}:/maro/maro/rl",
                f"{maro_comm_dir}:/maro/maro/communication"
            ],
            "command": "python3 /maro/cim/dqn/async_mode/policy_server.py"
        }
    }
}

for i in range(num_actors):
    actor_id = f"ACTOR.{i}"
    actor_manifest = deepcopy(docker_compose_manifest["services"]["policy_server"])
    del actor_manifest["build"]
    actor_manifest["command"] = "python3 /maro/cim/dqn/async_mode/actor.py"
    actor_manifest["container_name"] = actor_id
    actor_manifest["environment"] = [f"ACTORID={actor_id}"]
    docker_compose_manifest["services"][actor_id] = actor_manifest

if config["policy_manager"]["policy_training_mode"] == "multi-node":
    for i in range(num_trainers):
        trainer_id = f"TRAINER.{i}"
        trainer_manifest = deepcopy(docker_compose_manifest["services"]["policy_server"])
        del trainer_manifest["build"]
        trainer_manifest["command"] = "python3 /maro/cim/dqn/async_mode/trainer.py"
        trainer_manifest["container_name"] = trainer_id
        trainer_manifest["environment"] = [f"TRAINERID={trainer_id}"]
        docker_compose_manifest["services"][trainer_id] = trainer_manifest

with open(join(cim_dqn_async_dir, "docker-compose.yml"), "w") as fp:
    yaml.safe_dump(docker_compose_manifest, fp)
