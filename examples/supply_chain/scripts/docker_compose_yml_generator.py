# version: "3.9"
# services:
#   redis:
#     image: redis:6
#     container_name: maro-redis
#   learner:
#     build:
#       context: .
#       dockerfile: docker_files/dev.df
#     image: maro-dev
#     container_name: learner
#     volumes:
#       - /home/data_disk/yaqiu/maro/examples:/maro/examples
#     command: ["python3", "/maro/examples/supply_chain/dqn/distributed_launcher.py", "-w", "1"]
#   actor_1:
#     image: maro-dev
#     container_name: actor_1
#     volumes:
#       - /home/data_disk/yaqiu/maro/examples:/maro/examples
#     command: ["python3", "/maro/examples/supply_chain/dqn/distributed_launcher.py", "-w", "2"]
#   actor_2:
#     image: maro-dev
#     container_name: actor_2
#     volumes:
#       - /home/data_disk/yaqiu/maro/examples:/maro/examples
#     command: ["python3", "/maro/examples/supply_chain/dqn/distributed_launcher.py", "-w", "2"]

import defaultdict
import yaml
from os.path import dirname, join, realpath

path = realpath(__file__)
config_path = join(dirname(dirname(path)), "dqn", "config.yml")


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
            "volumes": ["/home/data_disk/yaqiu/maro/examples:/maro/examples"]
        }
    }
}


build:
#       context: .
#       dockerfile: docker_files/dev.df
#     image: maro-dev
#     container_name: learner
#     volumes:
#       - /home/data_disk/yaqiu/maro/examples:/maro/examples
#     command: ["python3", "/maro/examples/supply_chain/dqn/distributed_launcher.py", "-w", "1"]