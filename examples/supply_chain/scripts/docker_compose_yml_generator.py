import defaultdict
import yaml
from os import makedirs
from os.path import dirname, join, realpath

path = realpath(__file__)
script_dir = dirname(path)
sc_code_dir = dirname(script_dir)
config_path = join(sc_code_dir, "dqn", "config.yml")

with open(config_path, "r") as fp:
    config = yaml.safe_load(fp)
    num_actors = config["distributed"]["num_actors"]

docker_compose_yaml = {
    "version": "3.9", 
    "services": {
        "redis": {"image": "redis:6", "container_name": "maro-redis"},
        "learner": {
            "build": {"context": ".", "dockerfile": "docker_files/dev.df"},
            "image": "maro-dev",
            "container_name": "learner",
            "volumes": [f"{sc_code_dir}:/maro/supply_chain"],
            "command": ["python3", "/maro/supply_chain/dqn/distributed_launcher.py", "-w", "1"]
        }
    }
}

actor_template = docker_compose_yaml["services"]["learner"].copy()
del actor_template["build"]
actor_template["command"][-1] = "2"

for i in range(num_actors):
    actor_id = f"actor_{i}"
    actor_template["container_name"] = actor_id
    docker_compose_yaml["services"][actor_id] = actor_template

docker_compose_yaml_dir = join(sc_code_dir, "docker_compose_yamls")
makedirs(docker_compose_yaml_dir, exist_ok=True)
with open(join(docker_compose_yaml_dir, "docker-compose.yml"), "w") as fp:
    yaml.safe_dump(docker_compose_yaml, fp)
