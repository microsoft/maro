# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os
import subprocess

import redis
import yaml

from maro.utils.exception.cli_exception import CliException

LOCAL_PROCESS_PATH = os.path.expanduser("~/.maro/local")


def load_details(deployment_path: str = None):
    try:
        with open(deployment_path, "r") as cf:
            details = yaml.safe_load(cf)
    except Exception as e:
        raise CliException(f"Failure to find job details, cause by {e}")

    return details


def load_redis_info():
    with open(os.path.join(LOCAL_PROCESS_PATH, "redis_info.yml"), "r") as rf:
        redis_info = yaml.safe_load(rf)

    return redis_info


def env_preset():
    """Need Redis ready and master agent start."""
    if not os.path.exists(LOCAL_PROCESS_PATH):
        os.makedirs(LOCAL_PROCESS_PATH)

    if "redis_info.yml" in os.listdir(LOCAL_PROCESS_PATH):
        redis_info = load_redis_info()

        redis_connection = redis.Redis(host=redis_info["host"], port=redis_info["port"])
        redis_connection.hset("local_process:setting", "redis_pid", redis_info["pid"])
    else:
        # create redis by random port, write down to ~/.maro/local
        redis_connection = start_redis()

    agent_status = redis_connection.hget("local_process:setting", "agent_status")
    if not agent_status:
        start_agent(redis_connection)

    return redis_connection


def start_redis():
    import socket

    # Get random free port.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as temp_socket:
        temp_socket.bind(("", 0))
        random_port = temp_socket.getsockname()[1]

    redis_process = subprocess.Popen(["redis-server", "--port", str(random_port), "--daemonize yes"])
    redis_process.wait(timeout=2)

    redis_info = {"host": "localhost", "port": int(random_port), "pid": redis_process.pid}

    with open(os.path.join(LOCAL_PROCESS_PATH, "redis_info.yml"), "w") as wf:
        yaml.safe_dump(redis_info, wf)

    redis_connection = redis.Redis(host="localhost", port=int(random_port))
    redis_connection.hset("local_process:setting", "redis_pid", redis_info["pid"])

    return redis_connection


def start_agent(redis_connection):
    # get agent.py path
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    agent_file_path = os.path.join(current_file_path, "..", "agent/job_agent.py")

    command = f"python {agent_file_path}"
    agent_process = subprocess.Popen(command, shell=True)
    redis_connection.hset("local_process:setting", "agent_status", 1)
    redis_connection.hset("local_process:setting", "agent_pid", agent_process.pid)
