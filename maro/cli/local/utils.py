# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import subprocess
from copy import deepcopy
from typing import Dict, List

import docker
import yaml

from maro.cli.utils.common import format_env_vars


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
            proc.stop(timeout=timeout)
            proc.remove()

        client = docker.from_env()
        job_network = client.networks.get(job_name)
        job_network.remove()


def exec(cmd: str, env: dict, debug: bool = False) -> subprocess.Popen:
    stream = None if debug else subprocess.PIPE
    return subprocess.Popen(
        cmd.split(), env={**os.environ.copy(), **env}, stdout=stream, stderr=stream, encoding="utf8"
    )


def start_rl_job(env_by_component: Dict[str, dict], maro_root: str, background: bool = False) -> List[subprocess.Popen]:
    def get_local_script_path(component: str):
        return os.path.join(maro_root, "maro", "rl", "workflows", f"{component.split('-')[0]}.py")

    procs = [
        exec(
            f"python {get_local_script_path(component)}",
            format_env_vars({**env, "PYTHONPATH": maro_root}, mode="proc"),
            debug=not background
        )
        for component, env in env_by_component.items()
    ]
    if not background:
        for proc in procs:
            proc.communicate()

    return procs


def start_rl_job_in_containers(
    conf: dict, image_name: str, env_by_component: Dict[str, dict], path_mapping: Dict[str, str]
) -> None:
    job_name = conf["job"]
    client, containers = docker.from_env(), []
    if conf["training"]["mode"] != "simple" or conf["rollout"].get("parallelism", 1) > 1:
        # create the exclusive network for the job
        client.networks.create(job_name, driver="bridge")

    for component, env in env_by_component.items():
        container_name = f"{job_name}.{component}"
        # volume mounts for scenario folder, policy loading, checkpointing and logging
        container = client.containers.run(
            image_name,
            command=f"python3 /maro/maro/rl/workflows/{component.split('-')[0]}.py",
            detach=True,
            name=container_name,
            environment=env,
            volumes=[f"{src}:{dst}" for src, dst in path_mapping.items()] + ["/home/yaqiu/maro/maro/rl:/maro/maro/rl"],
            network=job_name
        )

        containers.append(container)

    return containers


def get_docker_compose_yml_path() -> str:
    return os.path.join(os.getcwd(), "docker-compose.yml")


def start_rl_job_with_docker_compose(
    conf: dict, context: str, dockerfile_path: str, image_name: str, env_by_component: Dict[str, dict],
    path_mapping: Dict[str, str]
) -> None:
    common_spec = {
        "build": {"context": context, "dockerfile": dockerfile_path},
        "image": image_name,
        "volumes": [f"{src}:{dst}" for src, dst in path_mapping.items()] + ["/home/yaqiu/maro/maro/rl:/maro/maro/rl"]
    }
    job = conf["job"]
    manifest = {"version": "3.9"}
    manifest["services"] = {
        component: {
            **deepcopy(common_spec),
            **{
                "container_name": f"{job}.{component}",
                "command": f"python3 /maro/maro/rl/workflows/{component.split('-')[0]}.py",
                "environment": format_env_vars(env, mode="docker-compose")
            }
        }
        for component, env in env_by_component.items()
    }

    with open(get_docker_compose_yml_path(), "w") as fp:
        yaml.safe_dump(manifest, fp)

    subprocess.run(["docker-compose", "--project-name", job, "up", "--remove-orphans"])


def stop_rl_job_with_docker_compose(job_name: str):
    subprocess.run(["docker-compose", "--project-name", job_name, "down"])
    os.remove(get_docker_compose_yml_path())
