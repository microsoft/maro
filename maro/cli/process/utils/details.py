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
from maro.utils.exception.cli_exception import CliException


def load_details(deployment_path: str = None):
    try:
        with open(deployment_path, "r") as cf:
            details = yaml.safe_load(cf)
    except Exception as e:
        raise CliException(f"Failure to find job details, cause by {e}")

    return details


def save_setting_info(setting_info):
    with open(os.path.expanduser(LocalPaths.MARO_PROCESS_SETTING), "w") as wf:
        yaml.safe_dump(setting_info, wf)


def load_redis_info():
    try:
        with open(os.path.expanduser(LocalPaths.MARO_PROCESS_SETTING), "r") as rf:
            redis_info = yaml.safe_load(rf)
    except Exception as e:
        raise CliException(
            f"Failure to load setting information, cause by {e}"
            f"Please run maro process setup, before any process commands."
        )

    return redis_info


def close_by_pid(pid: Union[int, list], recursize: bool = False):
    if isinstance(pid, int):
        current_process = psutil.Process(pid)
        if recursize:
            children_pid = get_child_pid(pid)
            # May launch by JobTrackingAgent which is child process, so need close parent process first.
            current_process.kill()
            for child_pid in children_pid:
                child_process = psutil.Process(child_pid)
                child_process.kill()
        else:
            current_process.kill()
    else:
        assert(not recursize)
        for p in pid:
            os.kill(p, signal.SIGKILL)


def env_preset():
    """Need Redis ready and master agent start."""
    setting_info = load_redis_info()

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


def get_child_pid(parent_pid):
    command = f"ps -o pid --ppid {parent_pid} --noheaders"
    children_pid_process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    children_pid = children_pid_process.stdout.read()

    try:
        children_pid = int(children_pid)
    except Exception:
        children_pid = children_pid.decode().split("\n")
        children_pid = [int(pid) for pid in children_pid[:-1]]

    return children_pid
