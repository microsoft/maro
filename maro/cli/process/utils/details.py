# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os
import redis
import subprocess
import yaml

from maro.cli.process.agent.job_agent import MasterAgent
from maro.utils.exception.cli_exception import CliException


LOCAL_PROCESS_PATH = os.path.expanduser("~/.maro/local")


def load_details(deployment_path: str = None):
    try:
        with open(deployment_path, "r") as cf:
            details = yaml.safe_load(cf)
    except Exception as e:
        raise CliException(f"Failure to find job details, cause by {e}")

    return details


def env_preset():
    """Need Redis ready and master agent start."""
    if not os.path.exists(LOCAL_PROCESS_PATH):
        os.makedirs(LOCAL_PROCESS_PATH)

    if "redis_info.yml" in os.listdir(LOCAL_PROCESS_PATH):
        with open(os.path.join(LOCAL_PROCESS_PATH, "redis_info.yml"), "r") as rf:
            redis_info = yaml.safe_load(rf)

        redis_connection = redis.Redis(host=redis_info["host"], port=redis_info["port"])
    else:
        # create redis by random port, write down to ~/.maro/local
        redis_connection = start_redis()

    agent_status = redis_connection.get("local_process:agent_status")
    if not agent_status:
        master_agent = MasterAgent(redis_info)
        master_agent.start()
        redis_connection.set("local_process:agent_status", 1)

    return redis_connection


def start_redis():
    import socket

    # Get random free port.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as temp_socket:
        temp_socket.bind(("", 0))
        random_port = temp_socket.getsockname()[1]

    redis_process = subprocess.Popen(["redis-server", "--port", str(random_port), "--daemonize yes"])

    with open(os.path.join(LOCAL_PROCESS_PATH, "redis_info.yml"), "w") as wf:
        yaml.safe_dump({"host": "localhost", "port": int(random_port)})

    return redis.Redis(host="localhost", port=int(random_port))
