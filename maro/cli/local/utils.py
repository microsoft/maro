# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import subprocess
from copy import deepcopy
from typing import List

import docker
import yaml

from maro.cli.utils.common import format_env_vars
from maro.rl.workflows.config.parser import ConfigParser


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


def stop_redis(port: int):
    subprocess.Popen(["redis-cli", "-p", str(port), "shutdown"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


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
            try:
                proc.stop(timeout=timeout)
                proc.remove()
            except Exception:
                pass

        client = docker.from_env()
        try:
            job_network = client.networks.get(job_name)
            job_network.remove()
        except Exception:
            pass


def exec(cmd: str, env: dict, debug: bool = False) -> subprocess.Popen:
    stream = None if debug else subprocess.PIPE
    return subprocess.Popen(
        cmd.split(),
        env={**os.environ.copy(), **env},
        stdout=stream,
        stderr=stream,
        encoding="utf8",
    )


def start_rl_job(
    parser: ConfigParser,
    maro_root: str,
    evaluate_only: bool,
    background: bool = False,
) -> List[subprocess.Popen]:
    procs = [
        exec(
            f"python {script}" + ("" if not evaluate_only else " --evaluate_only"),
            format_env_vars({**env, "PYTHONPATH": maro_root}, mode="proc"),
            debug=not background,
        )
        for script, env in parser.get_job_spec().values()
    ]
    if not background:
        for proc in procs:
            proc.communicate()

    return procs


def start_rl_job_in_containers(parser: ConfigParser, image_name: str) -> list:
    job_name = parser.config["job"]
    client, containers = docker.from_env(), []
    training_mode = parser.config["training"]["mode"]
    if "parallelism" in parser.config["rollout"]:
        rollout_parallelism = max(
            parser.config["rollout"]["parallelism"]["sampling"],
            parser.config["rollout"]["parallelism"].get("eval", 1),
        )
    else:
        rollout_parallelism = 1
    if training_mode != "simple" or rollout_parallelism > 1:
        # create the exclusive network for the job
        client.networks.create(job_name, driver="bridge")

    for component, (script, env) in parser.get_job_spec(containerize=True).items():
        # volume mounts for scenario folder, policy loading, checkpointing and logging
        container = client.containers.run(
            image_name,
            command=f"python3 {script}",
            detach=True,
            name=component,
            environment=env,
            volumes=[f"{src}:{dst}" for src, dst in parser.get_path_mapping(containerize=True).items()],
            network=job_name,
        )

        containers.append(container)

    return containers


def get_docker_compose_yml_path(maro_root: str) -> str:
    return os.path.join(maro_root, ".tmp", "docker-compose.yml")


def start_rl_job_with_docker_compose(
    parser: ConfigParser,
    context: str,
    dockerfile_path: str,
    image_name: str,
    evaluate_only: bool,
) -> None:
    common_spec = {
        "build": {"context": context, "dockerfile": dockerfile_path},
        "image": image_name,
        "volumes": [f"./{src}:{dst}" for src, dst in parser.get_path_mapping(containerize=True).items()],
    }

    job_name = parser.config["job"]
    manifest = {
        "version": "3.9",
        "services": {
            component: {
                **deepcopy(common_spec),
                **{
                    "container_name": component,
                    "command": f"python3 {script}" + ("" if not evaluate_only else " --evaluate_only"),
                    "environment": format_env_vars(env, mode="docker-compose"),
                },
            }
            for component, (script, env) in parser.get_job_spec(containerize=True).items()
        },
    }

    docker_compose_file_path = get_docker_compose_yml_path(maro_root=context)
    with open(docker_compose_file_path, "w") as fp:
        yaml.safe_dump(manifest, fp)

    subprocess.run(
        ["docker-compose", "--project-name", job_name, "-f", docker_compose_file_path, "up", "--remove-orphans"],
    )


def stop_rl_job_with_docker_compose(job_name: str, context: str):
    subprocess.run(["docker-compose", "--project-name", job_name, "down"])
    os.remove(get_docker_compose_yml_path(maro_root=context))
