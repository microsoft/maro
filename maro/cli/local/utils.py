# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import subprocess
from copy import deepcopy

import docker
import yaml

from maro.cli.utils import config_parser


class RedisHashKey:
    """Record Redis elements name, and only for maro process"""
    JOB_CONF = "job_conf"
    JOB_DETAILS = "job_details"


class JobStatus:
    PENDING = "pending"
    IMAGE_BUILDING = "image_building"
    RUNNING = "running"
    ERROR = "error"
    REMOVED = "removed"
    FINISHED = "finished"


def start_redis(port: int):
    subprocess.Popen(["redis-server", "--port", str(port)], stdout=subprocess.DEVNULL)


def start_redis_container(port: int, name: str, network: str):
    # create the exclusive network for containerized job management
    client = docker.from_env()
    client.networks.create(network, driver="bridge")
    client.containers.run("redis", network=network, name=name, detach=True, ports={"6379/tcp": ("127.0.0.1", port)})


def stop_redis(port: int):
    subprocess.Popen(["redis-cli", "-p", str(port), "shutdown"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def stop_redis_container(name: str, network: str, timeout: int = 5):
    client = docker.from_env()
    container = client.containers.get(name)
    container.stop(timeout=timeout)
    container.remove()
    network = client.networks.get(network)
    network.remove()


def extract_error_msg_from_docker_log(container: docker.models.containers.Container):
    logs = container.logs().decode().splitlines()
    for i, log in enumerate(logs):
        if "Traceback (most recent call last):" in log:
            return "\n".join(logs[i:])

    return logs


def check_proc_status(proc):
    if isinstance(proc, subprocess.Popen):
        if proc.poll() is None:
            return True, 0, None
        _, err_out = proc.communicate()
        return False, proc.returncode, err_out
    else:
        client = docker.from_env()
        container_state = client.api.inspect_container(proc.id)["State"]
        return container_state["Running"], container_state["ExitCode"], extract_error_msg_from_docker_log(proc)


def poll(procs):
    error, running = False, []
    for proc in procs:
        is_running, exit_code, err_out = check_proc_status(proc)
        if is_running:
            running.append(proc)
        elif exit_code:
            error = True
            break

    return error, err_out, running


def term(procs, job_name: str, timeout: int = 3):
    if isinstance(procs[0], subprocess.Popen):
        for proc in procs:
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
    else:
        for proc in procs:
            proc.stop(timeout=timeout)
            proc.remove()

        client = docker.from_env()
        job_network = client.networks.get(job_name)
        job_network.remove()


def exec(cmd: str, env: dict, debug: bool = False):
    stream = None if debug else subprocess.PIPE
    return subprocess.Popen(
        cmd.split(), env={**os.environ.copy(), **env}, stdout=stream, stderr=stream, encoding="utf8"
    )


def get_proc_specs(config: dict, redis_port: int, python_path: str):
    return {
        component: (
            config_parser.get_script_path(component),
            config_parser.format_env_vars(
                {**env, "REDIS_HOST": "localhost", "REDIS_PORT": str(redis_port), "PYTHONPATH": python_path},
                mode="proc"
            )
        )
        for component, env in config_parser.get_rl_component_env_vars(config).items()
    }


def get_container_specs(config: dict, redis_host: str):
    return {
        component: (
            config_parser.get_script_path(component, containerized=True),
            {**env, "REDIS_HOST": redis_host, "REDIS_PORT": "6379"}
        )
        for component, env in config_parser.get_rl_component_env_vars(config, containerized=True).items()
    }


def get_docker_compose_yml(config: dict, context: str, dockerfile_path: str, image_name: str):
    common_spec = {
        "build": {"context": context, "dockerfile": dockerfile_path},
        "image": image_name,
        "volumes": [
            f"{config['scenario_path']}:{config_parser.get_mnt_path_in_container('scenario')}",
            f"{config['log_path']}:{config_parser.get_mnt_path_in_container('logs')}",
            f"{config['checkpoint_path']}:{config_parser.get_mnt_path_in_container('checkpoints')}",
            f"{config['load_path']}:{config_parser.get_mnt_path_in_container('loadpoint')}",
            "/home/yaqiu/maro/maro/rl:/maro/maro/rl"
        ]
    }
    job = config["job"]
    # redis_host = f"{job}.redis"
    manifest = {"version": "3.9"}
    manifest["services"] = {
        component: {
            **deepcopy(common_spec),
            **{
                "container_name": f"{job}.{component}",
                "command": f"python3 {config_parser.get_script_path(component, containerized=True)}",
                "environment": config_parser.format_env_vars(env, mode="docker-compose")
            }
        }
        for component, env in config_parser.get_rl_component_env_vars(config, containerized=True).items()
    }

    return manifest


def get_docker_compose_yml_path():
    return os.path.join(os.getcwd(), "docker-compose.yml")


def start_rl_job_in_background(config: dict, redis_port: int, python_path: str):
    return [exec(f"python {cmd}", env) for cmd, env in get_proc_specs(config, redis_port, python_path).values()]


def start_rl_job_in_containers(config, image_name: str, redis_host: str, network: str = None):
    job_name = config["job"]
    client, containers = docker.from_env(), []
    if config["mode"] != "single":
        # create the exclusive network for the job
        job_network = client.networks.create(job_name, driver="bridge")

    for component, (cmd, env) in get_container_specs(config, redis_host).items():
        container_name = f"{job_name}.{component}"
        # volume mounts for scenario folder, policy loading, checkpointing and logging
        container = client.containers.run(
            image_name,
            command=f"python3 {cmd}",
            detach=True,
            name=container_name,
            environment=env,
            volumes=[
                f"{config['scenario_path']}:{config_parser.get_mnt_path_in_container('scenario')}",
                f"{config['log_path']}:{config_parser.get_mnt_path_in_container('logs')}",
                f"{config['checkpoint_path']}:{config_parser.get_mnt_path_in_container('checkpoints')}",
                f"{config['load_path']}:{config_parser.get_mnt_path_in_container('loadpoint')}",
            ],
            network=network if config["mode"] != "single" else None
        )

        # if completed_process.returncode:
        #     raise ResourceAllocationFailed(completed_process.stderr)
        if config["mode"] != "single":
            job_network.connect(container)
        containers.append(container)

    return containers


def start_rl_job_in_foreground(config: dict, python_path: str, port: int = 20000):
    procs = [exec(f"python {cmd}", env, debug=True) for cmd, env in get_proc_specs(config, port, python_path).values()]
    for proc in procs:
        proc.communicate()


def start_rl_job_with_docker_compose(config: dict, context: str, dockerfile_path: str, image_name: str):
    manifest = get_docker_compose_yml(config, context, dockerfile_path, image_name)
    with open(get_docker_compose_yml_path(), "w") as fp:
        yaml.safe_dump(manifest, fp)

    subprocess.run(["docker-compose", "--project-name", config["job"], "up", "--remove-orphans"])


def stop_rl_job_with_docker_compose(config):
    subprocess.run(["docker-compose", "--project-name", config["job"], "down"])
    os.remove(get_docker_compose_yml_path())
