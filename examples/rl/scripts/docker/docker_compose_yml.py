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
    example_dir = dirname(dirname(docker_script_dir))
    root_dir = dirname(dirname(example_dir))
    maro_rl_dir = join(root_dir, "maro", "rl")
    sc_dir = join(root_dir, "maro", "simulator", "scenarios", "supply_chain")

    with open(join(example_dir, "config.yml"), "r") as fp:
        config = yaml.safe_load(fp)

    common_spec = {
        "build": {"context": root_dir, "dockerfile": join(root_dir, "docker_files", "dev.df")},
        "image": "marorl",
        "volumes": [
            f"{example_dir}:/maro/examples",
            f"{maro_rl_dir}:/maro/maro/rl",
            f"{sc_dir}:/maro/maro/simulator/scenarios/supply_chain"
        ]
    }

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

    if "scripts" in config and "grad_worker" in config["scripts"]:
        grad_worker_path = config["scripts"]["grad_worker"]
    else:
        grad_worker_path = join(workflow_dir_in_container, "grad_worker.py")

    if config["mode"] == "single":
        envs = [
            f"JOB={config['job']}",
            f"SCENARIODIR={config['scenario_dir']}",
            f"SCENARIO={config['scenario']}",
            f"MODE=single",
            f"NUMEPISODES={config['num_episodes']}",
            f"EVALSCH={config['eval_schedule']}"
        ]
        if "num_steps" in config:
            envs.append(f"NUMSTEPS={config['num_steps']}")
        if "load_policy_dir" in config:
            envs.append(f"LOADDIR={config['load_policy_dir']}")
        if "checkpoint_dir" in config:
            envs.append(f"CHECKPOINTDIR={config['checkpoint_dir']}")
        if "log_dir" in config:
            envs.append(f"LOGDIR={config['log_dir']}")

        docker_compose_manifest = {"services": {
            "main": {
                **common_spec,
                **{
                    "container_name": f"{namespace}.main",
                    "command": f"python3 {main_path}",
                    "environment": envs
                }
            }
        }}
    else:
        redis_host = config["redis"]["host"]
        docker_compose_manifest = {
            "version": "3.9",
            "services": {"redis": {"image": "redis:6", "container_name": f"{namespace}.{redis_host}"}}
        }

        policy_group = "-".join([config['job'], 'policies'])
        common_envs = [
            f"REDISHOST={namespace}.{redis_host}",
            f"REDISPORT={config['redis']['port']}",
            f"JOB={config['job']}",
            f"SCENARIODIR={config['scenario_dir']}",
            f"SCENARIO={config['scenario']}",
            f"MODE={config['mode']}",
            f"POLICYMANAGERTYPE={config['policy_manager']['type']}"
        ]

        if "load_policy_dir" in config:
            common_envs.append(f"LOADDIR={config['load_policy_dir']}")
        if "checkpoint_dir" in config:
            common_envs.append(f"CHECKPOINTDIR={config['checkpoint_dir']}")
        if "log_dir" in config:
            common_envs.append(f"LOGDIR={config['log_dir']}")
        if "data_parallel" in config:
            common_envs.append(f"DATAPARALLEL=1")
            common_envs.append(f"NUMGRADWORKERS={config['data_parallel']['num_workers']}")
            common_envs.append(f"ALLOCATIONMODE={config['data_parallel']['allocation_mode']}")
        if config["policy_manager"]["type"] == "distributed":
            common_envs.append(f"POLICYGROUP={policy_group}")
            common_envs.append(f"NUMHOSTS={config['policy_manager']['distributed']['num_hosts']}")

        # grad worker config
        if "data_parallel" in config:
            for worker_id in range(config['data_parallel']['num_workers']):
                str_id = f"grad_worker.{worker_id}"
                grad_worker_spec = deepcopy(common_spec)
                del grad_worker_spec["build"]
                grad_worker_spec["command"] = f"python3 {grad_worker_path}"
                grad_worker_spec["container_name"] = f"{namespace}.{str_id}"
                grad_worker_spec["environment"] = [f"WORKERID={worker_id}"] + common_envs
                docker_compose_manifest["services"][str_id] = grad_worker_spec

        # host spec
        if config["policy_manager"]["type"] == "distributed":
            for host_id in range(config["policy_manager"]["distributed"]["num_hosts"]):
                str_id = f"policy_host.{host_id}"
                host_spec = deepcopy(common_spec)
                del host_spec["build"]
                host_spec["command"] = f"python3 {policy_host_path}"
                host_spec["container_name"] = f"{namespace}.{str_id}"
                host_spec["environment"] = [f"HOSTID={host_id}"] + common_envs
                docker_compose_manifest["services"][str_id] = host_spec

        mode = config["mode"]
        if mode == "sync":
            # main process spec
            rollout_group = "-".join([config['job'], 'rollout'])
            envs = [
                f"ROLLOUTTYPE={config['sync']['rollout_type']}",
                f"NUMEPISODES={config['num_episodes']}",
                f"NUMROLLOUTS={config['sync']['num_rollouts']}"
            ]
            if "num_steps" in config:
                envs.append(f"NUMSTEPS={config['num_steps']}")
            if "eval_schedule" in config:
                envs.append(f"EVALSCH={config['eval_schedule']}")
            if config["sync"]["rollout_type"] == "distributed":
                envs.append(f"ROLLOUTGROUP={rollout_group}")
                if "min_finished_workers" in config["sync"]["distributed"]:
                    envs.append(f"MINFINISH={config['sync']['distributed']['min_finished_workers']}")
                if "max_extra_recv_tries" in config["sync"]["distributed"]:
                    envs.append(f"MAXEXRECV={config['sync']['distributed']['max_extra_recv_tries']}")
                if "extra_recv_timeout" in config["sync"]["distributed"]:
                    envs.append(f"MAXRECVTIMEO={config['sync']['distributed']['extra_recv_timeout']}")

            if "num_eval_rollouts" in config["sync"]:
                envs.append(f"NUMEVALROLLOUTS={config['sync']['num_eval_rollouts']}")

            docker_compose_manifest["services"]["main"] = {
                **common_spec,
                **{
                    "container_name": f"{namespace}.main",
                    "command": f"python3 {main_path}",
                    "environment": envs + common_envs
                }
            }
            # rollout worker spec
            if config["sync"]["rollout_type"] == "distributed":
                for worker_id in range(config["sync"]["num_rollouts"]):
                    str_id = f"rollout_worker.{worker_id}"
                    worker_spec = deepcopy(common_spec)
                    del worker_spec["build"]
                    worker_spec["command"] = f"python3 {rollout_worker_path}"
                    worker_spec["container_name"] = f"{namespace}.{str_id}"
                    worker_spec["environment"] = [f"WORKERID={worker_id}", f"ROLLOUTGROUP={rollout_group}"] + common_envs
                    docker_compose_manifest["services"][str_id] = worker_spec
        elif mode == "async":
            # policy server spec
            envs = [
                f"GROUP={config['job']}",
                f"NUMROLLOUTS={config['async']['num_rollouts']}"
            ]
            if "max_lag" in config["async"]:
                envs.append(f"MAXLAG={config['async']['max_lag']}")
            docker_compose_manifest["services"]["policy_server"] = {
                **common_spec,
                **{
                    "container_name": f"{namespace}.policy_server",
                    "command": f"python3 {policy_server_path}",
                    "environment": envs + common_envs
                }
            }
            # actor spec
            for actor_id in range(config["async"]["num_rollouts"]):
                str_id = f"actor.{actor_id}"
                actor_spec = deepcopy(common_spec)
                del actor_spec["build"]
                actor_spec["command"] = f"python3 {actor_path}"
                actor_spec["container_name"] = f"{namespace}.{str_id}"
                actor_spec["environment"] = [
                    f"ACTORID={actor_id}",
                    f"GROUP={config['job']}",
                    f"NUMEPISODES={config['num_episodes']}"
                ] + common_envs
                if "num_steps" in config:
                    actor_spec["environment"].append(f"NUMSTEPS={config['num_steps']}")
                docker_compose_manifest["services"][str_id] = actor_spec
        else:
            raise ValueError(f"mode must be 'sync' or 'async', got {mode}")

    with open(join(docker_script_dir, "yq.yml"), "w") as fp:
        yaml.safe_dump(docker_compose_manifest, fp)
