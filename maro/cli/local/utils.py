# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import docker
import os
import subprocess
from copy import deepcopy

import yaml

from maro.cli.local.meta import (
    DEFAULT_DOCKER_FILE_PATH, DEFAULT_DOCKER_IMAGE_NAME, DEFAULT_DOCKER_NETWORK, DEFAULT_REDIS_CONTAINER_NAME
)
from maro.cli.utils.rl_spec_generator import (
    format_env_vars, get_rl_component_env_vars, get_script_path, get_volume_mapping
)
from maro.utils.utils import LOCAL_MARO_ROOT


def start_redis(port: int):
    subprocess.Popen(["redis-server", "--port", str(port)], stdout=subprocess.DEVNULL)


def start_redis_container(port: int, network: str = DEFAULT_DOCKER_NETWORK):
    # create the exclusive network for containerized job management
    client = docker.from_env()
    client.networks.create(network, driver="bridge")
    client.containers.run(
        "redis", network=network, name=DEFAULT_REDIS_CONTAINER_NAME, detach=True,
        ports={"6379/tcp": ("127.0.0.1", port)}
    )


def stop_redis(port: int):
    subprocess.Popen(["redis-cli", "-p", str(port), "shutdown"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def stop_redis_container(network: str = DEFAULT_DOCKER_NETWORK, stop_redis_timeout: int = 5):
    client = docker.from_env()
    container = client.containers.get(DEFAULT_REDIS_CONTAINER_NAME)
    container.stop(timeout=stop_redis_timeout)
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
    return subprocess.Popen(cmd.split(), env=env, stdout=stream, stderr=stream, encoding="utf8")


def image_exists():
    try:
        client = docker.from_env()
        client.images.get(DEFAULT_DOCKER_IMAGE_NAME)
        return True
    except docker.errors.ImageNotFound:
        return False


def build_image():
    client = docker.from_env()
    with open(DEFAULT_DOCKER_FILE_PATH, "r") as fp:
        client.images.build(
            fileobj=fp,
            tag=DEFAULT_DOCKER_IMAGE_NAME,
            rm=True,
            custom_context=True,
        )


def get_proc_specs(config, redis_port: int):
    return {
        component: (
            get_script_path(component),
            format_env_vars(
                {**env, "REDISHOST": "localhost", "REDISPORT": str(redis_port), "PYTHONPATH": LOCAL_MARO_ROOT},
                mode="proc"
            )
        )
        for component, env in get_rl_component_env_vars(config).items()
    }


def get_container_specs(config):
    return {
        component: (
            get_script_path(component, containerized=True),
            {**env, "REDISHOST": DEFAULT_REDIS_CONTAINER_NAME, "REDISPORT": "6379"}
        )
        for component, env in get_rl_component_env_vars(config, containerized=True).items()
    }


def get_docker_compose_yml(config: dict):
    common_spec = {
        "build": {"context": LOCAL_MARO_ROOT, "dockerfile": DEFAULT_DOCKER_FILE_PATH},
        "image": DEFAULT_DOCKER_IMAGE_NAME,
        "volumes": [f"{local_pth}:{pth_in_cntr}" for local_pth, pth_in_cntr in get_volume_mapping(config).items()]
    }
    job = config["job"]
    redis_host = f"{job}.redis"
    manifest = {"version": "3.9"}
    manifest["services"] = {
        component: {
            **deepcopy(common_spec),
            **{
                "container_name": f"{job}.{component}",
                "command": f"python3 {get_script_path(component, containerized=True)}",
                "environment": format_env_vars(
                    {**env, "REDISHOST": redis_host, "REDISPORT": "6379"},
                    mode="docker-compose"
                )
            }
        }
        for component, env in get_rl_component_env_vars(config, containerized=True).items()
    }
    if config["mode"] != "single":
        manifest["services"]["redis"] = {"image": "redis", "container_name": redis_host}

    return manifest


def get_docker_compose_yml_path():
    return os.path.join(os.getcwd(), "docker-compose.yml")


def start_rl_job_in_background(config, redis_port: int):
    return [exec(f"python {cmd}", env) for cmd, env in get_proc_specs(config, redis_port).values()]


def start_rl_job_in_containers(config):
    job_name = config["job"]
    client, containers = docker.from_env(), []
    if config["mode"] != "single":
        # create the exclusive network for the job
        job_network = client.networks.create(job_name, driver="bridge")

    for component, (cmd, env) in get_container_specs(config).items():
        container_name = f"{job_name}.{component}"
        # volume mounts for scenario folder, policy loading, checkpointing and logging
        container = client.containers.run(
            DEFAULT_DOCKER_IMAGE_NAME,
            command=f"python3 {cmd}",
            detach=True,
            name=container_name,
            environment=env,
            volumes=[f"{src}:{target}" for src, target in get_volume_mapping(config).items()],
            network=DEFAULT_DOCKER_NETWORK if config["mode"] != "single" else None
        )

        # if completed_process.returncode:
        #     raise ResourceAllocationFailed(completed_process.stderr)
        if config["mode"] != "single":
            job_network.connect(container)
        containers.append(container)

    return containers


def start_rl_job_in_foreground(config, port: int = 20000):
    if config["mode"] != "single":
        start_redis(port)
    procs = [exec(f"python {cmd}", env, debug=True) for cmd, env in get_proc_specs(config, port).values()]
    for proc in procs:
        proc.communicate()
    if config["mode"] != "single":
        stop_redis(port)


def start_rl_job_with_docker_compose(config):
    manifest = get_docker_compose_yml(config)
    with open(get_docker_compose_yml_path(), "w") as fp:
        yaml.safe_dump(manifest, fp)

    subprocess.run(["docker-compose", "--project-name", config["job"], "up", "--remove-orphans"])


def stop_rl_job_with_docker_compose(config):
    subprocess.run(["docker-compose", "--project-name", config["job"], "down"])
    os.remove(get_docker_compose_yml_path())
