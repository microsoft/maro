# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import subprocess

import redis

from maro.cli.process.utils.default_param import process_setting
from maro.cli.process.utils.details import load_details, save_setting_info
from maro.cli.utils.params import LocalPaths, ProcessRedisName


def setup(deployment_path: str, **kwargs):
    current_process_path = os.path.expanduser(LocalPaths.MARO_PROCESS)
    # Create folder
    if not os.path.exists(current_process_path):
        os.makedirs(current_process_path)

    setting_info = process_setting
    if deployment_path is not None:
        customized_details = load_details(deployment_path=deployment_path)
        for key, value in customized_details.items():
            if key in setting_info:
                setting_info[key] = value

    save_setting_info(setting_info)

    # Start Redis
    redis_process = subprocess.Popen(
        ["redis-server", "--port", str(setting_info["redis_info"]["port"]), "--daemonize yes"]
    )
    redis_process.wait(timeout=2)

    redis_connection = redis.Redis(host=setting_info["redis_info"]["host"], port=setting_info["redis_info"]["port"])

    # Start agents
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    agent_file_path = os.path.join(current_file_path, "agent/job_agent.py")

    command = f"python {agent_file_path}"
    _ = subprocess.Popen(command, shell=True)
    redis_connection.hset(ProcessRedisName.SETTING, "agent_status", 1)

    # Push default setting into Redis
    del setting_info["redis_info"]
    redis_connection.hmset(ProcessRedisName.SETTING, setting_info)
