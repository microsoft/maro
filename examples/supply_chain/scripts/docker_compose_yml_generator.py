import yaml
from copy import deepcopy
from os import makedirs
from os.path import dirname, join, realpath

path = realpath(__file__)
script_dir = dirname(path)
sc_code_dir = dirname(script_dir)
root_dir = dirname(dirname(sc_code_dir))
config_path = join(sc_code_dir, "dqn", "config.yml")
dockerfile_path = join(root_dir, "docker_files", "dev.df")

with open(config_path, "r") as fp:
    config = yaml.safe_load(fp)
    num_actors = config["distributed"]["num_actors"]
    redis_host = config["distributed"]["redis_host"]

docker_compose_yaml = {
    "version": "3.9", 
    "services": {
        "redis": {"image": "redis:6", "container_name": redis_host},
        "learner": {
            "build": {"context": root_dir, "dockerfile": dockerfile_path},
            "image": "maro-sc",
            "container_name": "learner",
            "volumes": [f"{sc_code_dir}:/maro/supply_chain"],
            "command": ["python3", "/maro/supply_chain/dqn/distributed_launcher.py", "-w", "1"]
        }
    }
}

for i in range(num_actors):
    actor_id = f"actor_{i}"
    actor_template = deepcopy(docker_compose_yaml["services"]["learner"])
    del actor_template["build"]
    actor_template["command"][-1] = "2"
    actor_template["container_name"] = actor_id
    docker_compose_yaml["services"][actor_id] = actor_template

with open(join(sc_code_dir, "docker-compose.yml"), "w") as fp:
    yaml.safe_dump(docker_compose_yaml, fp)
