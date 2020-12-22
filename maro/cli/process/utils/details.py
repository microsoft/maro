# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import signal
import subprocess
from typing import Union

import psutil
import redis
import yaml

from maro.cli.utils.params import LocalPaths, ProcessRedisName
from maro.utils.exception.cli_exception import ProcessInternalError


def load_details(deployment_path: str = None):
    try:
        with open(deployment_path, "r") as cf:
            details = yaml.safe_load(cf)
    except Exception as e:
        raise ProcessInternalError(f"Failure to find job details, cause by {e}")

    return details


def load_setting_info():
    try:
        with open(os.path.expanduser(LocalPaths.MARO_PROCESS_SETTING), "r") as rf:
            redis_info = yaml.safe_load(rf)
    except Exception as e:
        raise ProcessInternalError(
            f"Failure to load setting information, cause by {e}"
            f"Please run maro process setup, before any process commands."
        )

    return redis_info


def save_setting_info(setting_info):
    with open(os.path.expanduser(LocalPaths.MARO_PROCESS_SETTING), "w") as wf:
        yaml.safe_dump(setting_info, wf)


def env_prepare():
    """Need Redis ready and master agent start."""
    setting_info = load_setting_info()

    redis_connection = redis.Redis(host=setting_info["redis_info"]["host"], port=setting_info["redis_info"]["port"])

    agent_status = int(redis_connection.hget(ProcessRedisName.SETTING, "agent_status"))
    if not agent_status:
        start_agent()
        redis_connection.hset(ProcessRedisName.SETTING, "agent_status", 1)

    return redis_connection


def start_agent():
    # start job_agent.py
    command = f"python {LocalPaths.MARO_PROCESS_AGENT}"
    _ = subprocess.Popen(command, shell=True)


def start_redis(port: int):
    # start Redis for maro
    redis_process = subprocess.Popen(
        ["redis-server", "--port", str(port), "--daemonize yes"]
    )
    redis_process.wait(timeout=2)


def close_by_pid(pid: Union[int, list], recursive: bool = False):
    if isinstance(pid, int):
        if not psutil.pid_exists(pid):
            return

        if recursive:
            current_process = psutil.Process(pid)
            children_process = current_process.children(recursive=False)
            # May launch by JobTrackingAgent which is child process, so need close parent process first.
            current_process.kill()
            for child_process in children_process:
                child_process.kill()
        else:
            os.kill(pid, signal.SIGKILL)
    else:
        for p in pid:
            if psutil.pid_exists(p):
                os.kill(p, signal.SIGKILL)


def get_child_pid(parent_pid):
    command = f"ps -o pid --ppid {parent_pid} --noheaders"
    get_children_pid_process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    children_pids = get_children_pid_process.stdout.read()
    get_children_pid_process.wait(timeout=2)

    # Convert into list or int
    try:
        children_pids = int(children_pids)
    except ValueError:
        children_pids = children_pids.decode().split("\n")
        children_pids = [int(pid) for pid in children_pids[:-1]]

    return children_pids
