# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import signal
import subprocess
from typing import Union

import psutil


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


def get_redis_pid_by_port(port: int):
    get_redis_pid_command = f"pidof 'redis-server *:{port}'"
    get_redis_pid_process = subprocess.Popen(get_redis_pid_command, shell=True, stdout=subprocess.PIPE)
    redis_pid = int(get_redis_pid_process.stdout.read())
    get_redis_pid_process.wait()

    return redis_pid
