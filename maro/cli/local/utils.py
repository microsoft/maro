# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
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
    subprocess.run(["docker", "network", "create", network], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(
        [
            "docker", "run", "-d",
            "--network", network,
            "--name", DEFAULT_REDIS_CONTAINER_NAME,
            "-p", f"127.0.0.1:{port}:6379/tcp",
            "redis"
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )


def stop_redis(port: int):
    subprocess.Popen(["redis-cli", "-p", str(port), "shutdown"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def stop_redis_container(network: str = DEFAULT_DOCKER_NETWORK, stop_redis_timeout: int = 5):
    subprocess.run(
        ["docker", "stop", "-t", str(stop_redis_timeout), DEFAULT_REDIS_CONTAINER_NAME],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    subprocess.run(["docker", "rm", DEFAULT_REDIS_CONTAINER_NAME], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(["docker", "network", "rm", network], stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def extract_error_msg_from_docker_log(container):
    logs = subprocess.run(["docker", "logs", container], stdout=subprocess.PIPE, encoding="utf8").stdout.splitlines()
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
        inspect = subprocess.run(
            ["docker", "inspect", proc, "--format={{.State.Running}} {{.State.ExitCode}}"],
            stdout=subprocess.PIPE, encoding="utf8"
        )
        is_running, exit_code = inspect.stdout.split()
        return is_running == "true", int(exit_code), extract_error_msg_from_docker_log(proc)


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


def term(procs, job_name, timeout: int = 3):
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
            subprocess.run(["docker", "stop", "-t", str(timeout), proc], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(["docker", "rm", proc], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if len(procs) > 1:
            subprocess.run(["docker", "network", "rm", job_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def exec(cmd: str, env: dict, debug: bool = False):
    stream = None if debug else subprocess.PIPE
    return subprocess.Popen(cmd.split(), env=env, stdout=stream, stderr=stream, encoding="utf8")


def image_exists():
    check = subprocess.run(
        ["docker", "image", "inspect", DEFAULT_DOCKER_IMAGE_NAME], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return bool(json.loads(check.stdout))


def build_image():
    subprocess.run(
        ["docker", "build", "--tag", DEFAULT_DOCKER_IMAGE_NAME, "-f", DEFAULT_DOCKER_FILE_PATH, LOCAL_MARO_ROOT],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
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
            format_env_vars(
                {**env, "REDISHOST": DEFAULT_REDIS_CONTAINER_NAME, "REDISPORT": "6379"}, mode="docker"
            )
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
    containers = []
    if config["mode"] != "single":
        # create the exclusive network for the job
        subprocess.run(["docker", "network", "create", job_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    for component, (cmd, env) in get_container_specs(config).items():
        container_name = f"{job_name}.{component}"
        run_container_cmd = ["docker", "run", "-it", "-d", "--name", container_name, *env]

        # volume mounts for scenario folder, policy loading, checkpointing and logging
        for local_path, path_in_cntr in get_volume_mapping(config).items():
            run_container_cmd.extend(["-v", f"{local_path}:{path_in_cntr}"])

        # A docker network is needed for distributed mode
        if config["mode"] != "single":
            run_container_cmd.extend(["--network", DEFAULT_DOCKER_NETWORK])

        run_container_cmd.extend([DEFAULT_DOCKER_IMAGE_NAME, "python3", cmd])
        subprocess.run(run_container_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # if completed_process.returncode:
        #     raise ResourceAllocationFailed(completed_process.stderr)
        if config["mode"] != "single":
            subprocess.run(
                ["docker", "network", "connect", job_name, container_name],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        containers.append(container_name)

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
