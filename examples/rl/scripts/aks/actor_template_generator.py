# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import yaml
from copy import deepcopy
from os.path import dirname, join, realpath

path = realpath(__file__)
docker_script_dir = dirname(path)
rl_example_dir = dirname(dirname(docker_script_dir))
root_dir = dirname(dirname(rl_example_dir))
workflow_dir = join(rl_example_dir, "workflows")
maro_rl_dir = join(root_dir, "maro", "rl")
maro_sc_dir = join(root_dir, "maro", "simulator", "scenarios", "supply_chain")
config_path = join(workflow_dir, "config.yml")
dockerfile_path = join(root_dir, "docker_files", "dev.df")

dir_path = dirname(realpath(__file__))
with open(join(dir_path, "learner.yml"), "r") as fp:
    mf = yaml.safe_load(fp)


container_spec = mf["spec"]["template"]["spec"]["containers"][0]
env_var = container_spec["env"]

for kv in env_var:
    if kv["name"] == "NUM_ACTORS":
        num_actors = int(kv["value"])

for i in range(num_actors):
    name = f"maro-actor-{i}"
    mf["metadata"]["name"] = name
    container_spec["name"] = name
    container_spec["command"][-1] = "2"
    with open(join(dir_path, f"actor_{i}.yml"), "w") as fp:
        yaml.safe_dump(mf, fp)
