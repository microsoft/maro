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
    path = realpath(__file__)
    docker_script_dir = dirname(path)
    rl_example_dir = dirname(dirname(docker_script_dir))
    root_dir = dirname(dirname(rl_example_dir))
    workflow_dir = join(rl_example_dir, "workflows")
    maro_rl_dir = join(root_dir, "maro", "rl")
    maro_comm_dir = join(root_dir, "maro", "communication")
    maro_sc_dir = join(root_dir, "maro", "simulator", "scenarios", "supply_chain")
    config_path = join(workflow_dir, "config.yml")
    dockerfile_path = join(root_dir, "docker_files", "dev.df")

    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)
        
    redis_host = config["redis"]["host"]
    docker_compose_manifest = {
        "version": "3.9",
        "services": {"redis": {"image": "redis:6", "container_name": f"{namespace}.{redis_host}"}}
    }
    common_spec = {
        "build": {"context": root_dir, "dockerfile": dockerfile_path},
        "image": "marorl",
        "volumes": [
            f"{rl_example_dir}:/maro/rl_examples",
            f"{maro_rl_dir}:/maro/maro/rl",
            f"{maro_comm_dir}:/maro/maro/communication",
            f"{maro_sc_dir}:/maro/maro/simulator/scenarios/supply_chain"
        ]
    }

    common_env = [
        f"REDISHOST={namespace}.{redis_host}",
        f"REDISPORT={config['redis']['port']}",
        f"JOB={config['job']}",
        f"SCENARIO={config['scenario']}",
        f"MODE={config['mode']}",
        f"POLICYMANAGERTYPE={config['policy_manager']['type']}",
        f"EXPDIST={'1' if config['rollout_experience_distribution'] else '0'}"
    ]

    if config["mode"] == "async":
        num_rollouts = config['async']['num_actors']
    elif config["sync"]["rollout_type"] == "simple":
        num_rollouts = config['sync']['simple']['parallelism']
    else:
        num_rollouts = config['sync']['distributed']['num_workers']

    common_env.append(f"NUMROLLOUTS={num_rollouts}")

    # host spec
    if config["policy_manager"]["type"] == "distributed":
        common_env.append(f"LEARNGROUP={config['policy_manager']['distributed']['group']}")
        common_env.append(f"NUMHOSTS={config['policy_manager']['distributed']['num_hosts']}")
        common_env.append(f"ALLOCATIONMODE={config['policy_manager']['distributed']['allocation_mode']}")
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
        # learner_spec
        docker_compose_manifest["services"]["learner"] = {
            **common_spec, 
            **{
                "container_name": f"{namespace}.learner",
                "command": "python3 /maro/rl_examples/workflows/learner.py",
                "environment": [
                    f"ROLLOUTTYPE={config['sync']['rollout_type']}",
                    f"EVALPARALLELISM={config['sync']['simple']['eval_parallelism']}",
                    f"ROLLOUTGROUP={config['sync']['distributed']['group']}",
                    f"NUMEPISODES={config['num_episodes']}",
                    f"EVALSCH={config['eval_schedule']}",
                    f"NUMEVALWORKERS={config['sync']['distributed']['num_eval_workers']}",
                    f"NUMSTEPS={config['num_steps']}",
                    f"MAXLAG={config['max_lag']}",
                    f"MINFINISH={config['sync']['distributed']['min_finished_workers']}",
                    f"MAXEXRECV={config['sync']['distributed']['max_extra_recv_tries']}",
                    f"MAXRECVTIMEO={config['sync']['distributed']['extra_recv_timeout']}",
                    f"PARALLEL={'1' if config['policy_manager']['simple']['parallel'] else '0'}"
                ] + common_env
            }
        }
        # rollout worker spec
        if config["sync"]["rollout_type"] == "distributed":
            for worker_id in range(config["sync"]["distributed"]["num_workers"]):
                str_id = f"rollout_worker.{worker_id}"
                worker_spec = deepcopy(common_spec)
                del worker_spec["build"]
                worker_spec["command"] = "python3 /maro/rl_examples/workflows/rollout.py"
                worker_spec["container_name"] = f"{namespace}.{str_id}"
                worker_spec["environment"] = [
                    f"WORKERID={worker_id}",
                    f"ROLLOUTGROUP={config['sync']['distributed']['group']}",
                    f"EVALSCH={config['eval_schedule']}"
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
                f"NUMSTEPS={config['num_steps']}",
                f"EVALSCH={config['eval_schedule']}"
            ] + common_env
            docker_compose_manifest["services"][str_id] = actor_spec
    else: 
        raise ValueError(f"mode must be 'sync' or 'async', got {mode}")

    with open(join(docker_script_dir, "yq.yml"), "w") as fp:
        yaml.safe_dump(docker_compose_manifest, fp)
