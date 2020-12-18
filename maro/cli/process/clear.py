# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import subprocess

import redis

from maro.cli.process.utils.details import close_by_pid, load_setting_info
from maro.cli.utils.params import LocalPaths, ProcessRedisName


def clear(**kwargs):
    setting_info = load_setting_info()

    # Build connection
    redis_connection = redis.Redis(host=setting_info["redis_info"]["host"], port=setting_info["redis_info"]["port"])

    # Stop running jobs
    running_jobs = redis_connection.hgetall(ProcessRedisName.RUNNING_JOB)
    if running_jobs:
        for job_name, pid_list in running_jobs.items():
            pid_list = json.loads(pid_list)
            close_by_pid(pid=pid_list, recursive=False)

    # Stop Agents
    agent_status = int(redis_connection.hget(ProcessRedisName.SETTING, "agent_status"))
    if agent_status:
        agent_pid = int(redis_connection.hget(ProcessRedisName.SETTING, "agent_pid"))
        close_by_pid(pid=agent_pid, recursive=True)
        redis_connection.hset(ProcessRedisName.SETTING, "agent_status", 0)

    # close Redis
    redis_mode = redis_connection.hget(ProcessRedisName.SETTING, "redis_mode").decode()
    if redis_mode == "MARO":
        get_redis_pid_command = f"pidof 'redis-server *:{setting_info['redis_info']['port']}'"
        get_redis_pid_process = subprocess.Popen(get_redis_pid_command, shell=True, stdout=subprocess.PIPE)
        redis_pid = int(get_redis_pid_process.stdout.read())
        get_redis_pid_process.wait()
        close_by_pid(pid=redis_pid, recursive=False)

    # Rm process environment setting
    os.remove(os.path.expanduser(LocalPaths.MARO_PROCESS_SETTING))
