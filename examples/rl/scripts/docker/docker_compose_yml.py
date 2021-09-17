# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import yaml
from copy import deepcopy
from os.path import dirname, join, realpath


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--namespace', help="job namespace", default="maro")
    args = parser.parse_args()

    namespace = args.namespace
    docker_script_dir = dirname(realpath(__file__))
    rl_example_dir = dirname(dirname(docker_script_dir))
    root_dir = dirname(dirname(rl_example_dir))
    workflow_dir = join(rl_example_dir, "workflows")
    maro_rl_dir = join(root_dir, "maro", "rl")

    with open(join(workflow_dir, "config.yml"), "r") as fp:
        config = yaml.safe_load(fp)

    common_spec = {
        "build": {"context": root_dir, "dockerfile": join(root_dir, "docker_files", "dev.df")},
        "image": "marorl",
        "volumes": [
            f"{rl_example_dir}:/maro/rl_examples",
            f"{maro_rl_dir}:/maro/maro/rl"
        ]
    }

    if config["mode"] == "single":
        docker_compose_manifest = {"services": {
            "main": {
                **common_spec, 
                **{
                    "container_name": f"{namespace}.main",
                    "command": "python3 /maro/rl_examples/workflows/learning_loop.py",
                    "environment": [
                        f"JOB={config['job']}",
                        f"SCENARIO={config['scenario']}",
                        f"MODE=single",
                        f"NUMEPISODES={config['num_episodes']}",
                        f"NUMSTEPS={config['num_steps']}",
                        f"EVALSCH={config['eval_schedule']}",
                    ]
                }
            }
        }}
    else:
        redis_host = config["redis"]["host"]
        docker_compose_manifest = {
            "version": "3.9",
            "services": {"redis": {"image": "redis:6", "container_name": f"{namespace}.{redis_host}"}}
        }

        common_env = [
            f"REDISHOST={namespace}.{redis_host}",
            f"REDISPORT={config['redis']['port']}",
            f"JOB={config['job']}",
            f"SCENARIO={config['scenario']}",
            f"MODE={config['mode']}",
            f"POLICYMANAGERTYPE={config['policy_manager']['type']}",
            f"CHECKPOINTDIR={config['checkpoint_dir']}"
        ]

        common_env.append(f"NUMROLLOUTS={config[config['mode']]['num_rollouts']}")
        common_env.append(f"DATAPARALLEL={config['data_parallel']['enable']}")
        common_env.append(f"DISTRIBUTED={config['policy_manager']['type'] == 'distributed'}")
        if config["data_parallel"]["enable"]:
            common_env.append(f"NUMGRADWORKERS={config['data_parallel']['num_workers']}")
            common_env.append(f"ALLOCATIONMODE={config['data_parallel']['allocation_mode']}")
        if config["policy_manager"]["type"] == "distributed":
            common_env.append(f"LEARNGROUP={config['policy_manager']['distributed']['group']}")
            common_env.append(f"NUMHOSTS={config['policy_manager']['distributed']['num_hosts']}")

        # grad worker config
        if config["data_parallel"]["enable"]:
            for worker_id in range(config['data_parallel']['num_workers']):
                str_id = f"grad_worker.{worker_id}"
                grad_worker_spec = deepcopy(common_spec)
                del grad_worker_spec["build"]
                grad_worker_spec["command"] = "python3 /maro/rl_examples/workflows/grad_worker.py"
                grad_worker_spec["container_name"] = f"{namespace}.{str_id}"
                grad_worker_spec["environment"] = [f"WORKERID={worker_id}"] + common_env
                docker_compose_manifest["services"][str_id] = grad_worker_spec

        # host spec
        if config["policy_manager"]["type"] == "distributed":
            for host_id in range(config["policy_manager"]["distributed"]["num_hosts"]):
                str_id = f"policy_host.{host_id}"
                host_spec = deepcopy(common_spec)
                del host_spec["build"]
                host_spec["command"] = "python3 /maro/rl_examples/workflows/policy_host.py"
                host_spec["container_name"] = f"{namespace}.{str_id}"
                host_spec["environment"] = [f"HOSTID={host_id}"] + common_env
                docker_compose_manifest["services"][str_id] = host_spec

        mode = config["mode"]
        if mode == "sync":
            # main process spec
            docker_compose_manifest["services"]["main"] = {
                **common_spec, 
                **{
                    "container_name": f"{namespace}.main",
                    "command": "python3 /maro/rl_examples/workflows/learning_loop.py",
                    "environment": [
                        f"ROLLOUTTYPE={config['sync']['rollout_type']}",
                        f"NUMEPISODES={config['num_episodes']}",
                        f"NUMSTEPS={config['num_steps']}",
                        f"EVALSCH={config['eval_schedule']}",
                        f"NUMEVALROLLOUTS={config[config['mode']]['num_eval_rollouts']}",
                        f"ROLLOUTGROUP={config['sync']['distributed']['group']}",
                        f"MAXLAG={config['max_lag']}",
                        f"MINFINISH={config['sync']['distributed']['min_finished_workers']}",
                        f"MAXEXRECV={config['sync']['distributed']['max_extra_recv_tries']}",
                        f"MAXRECVTIMEO={config['sync']['distributed']['extra_recv_timeout']}",
                    ] + common_env
                }
            }
            # rollout worker spec
            if config["sync"]["rollout_type"] == "distributed":
                for worker_id in range(config["sync"]["num_rollouts"]):
                    str_id = f"rollout_worker.{worker_id}"
                    worker_spec = deepcopy(common_spec)
                    del worker_spec["build"]
                    worker_spec["command"] = "python3 /maro/rl_examples/workflows/rollout.py"
                    worker_spec["container_name"] = f"{namespace}.{str_id}"
                    worker_spec["environment"] = [
                        f"WORKERID={worker_id}",
                        f"ROLLOUTGROUP={config['sync']['distributed']['group']}"
                    ] + common_env
                    docker_compose_manifest["services"][str_id] = worker_spec
        elif mode == "async":
            # policy server spec
            docker_compose_manifest["services"]["policy_server"] = {
                **common_spec, 
                **{
                    "container_name": f"{namespace}.policy_server",
                    "command": "python3 /maro/rl_examples/workflows/policy_manager.py",
                    "environment": [
                        f"GROUP={config['async']['group']}",
                        f"MAXLAG={config['max_lag']}"
                    ] + common_env
                }
            }
            # actor spec
            for actor_id in range(config["async"]["num_actors"]):
                str_id = f"actor.{actor_id}"
                actor_spec = deepcopy(common_spec)
                del actor_spec["build"]
                actor_spec["command"] = "python3 /maro/rl_examples/workflows/rollout.py"
                actor_spec["container_name"] = f"{namespace}.{str_id}"
                actor_spec["environment"] = [
                    f"ACTORID={actor_id}",
                    f"GROUP={config['async']['group']}",
                    f"NUMEPISODES={config['num_episodes']}",
                    f"NUMSTEPS={config['num_steps']}"
                ] + common_env
                docker_compose_manifest["services"][str_id] = actor_spec
        else: 
            raise ValueError(f"mode must be 'sync' or 'async', got {mode}")

    with open(join(docker_script_dir, "yq.yml"), "w") as fp:
        yaml.safe_dump(docker_compose_manifest, fp)
