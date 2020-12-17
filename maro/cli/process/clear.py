# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import subprocess

import redis

from maro.cli.process.utils.details import close_by_pid, load_redis_info
from maro.cli.utils.params import LocalPaths, ProcessRedisName


def clear(**kwargs):
    setting_info = load_redis_info()

    # Build connection
    redis_connection = redis.Redis(host=setting_info["redis_info"]["host"], port=setting_info["redis_info"]["port"])

    # Stop running jobs
    running_job = redis_connection.hgetall(ProcessRedisName.RUNNING_JOB)
    if running_job:
        for job_name, pid_dict in running_job.items():
            pid_dict = json.loads(pid_dict)
            for pids in pid_dict.values():
                close_by_pid(pid=pids, recursize=False)

    # Stop Agents
    agent_status = int(redis_connection.hget(ProcessRedisName.SETTING, "agent_status"))
    if agent_status:
        agent_pid = int(redis_connection.hget(ProcessRedisName.SETTING, "agent_pid"))
        close_by_pid(pid=agent_pid, recursize=True)
        redis_connection.hset(ProcessRedisName.SETTING, "agent_status", 0)

    # close Redis
    redis_mode = redis_connection.hget(ProcessRedisName.SETTING, "redis_mode").decode()
    if redis_mode == "MARO":
        get_redis_command = f"pidof 'redis-server *:{setting_info['redis_info']['port']}'"
        redis_process = subprocess.Popen(get_redis_command, shell=True, stdout=subprocess.PIPE)
        redis_pid = int(redis_process.stdout.read())
        redis_process.wait()
        close_by_pid(pid=redis_pid, recursize=False)

    os.remove(os.path.expanduser(LocalPaths.MARO_PROCESS_SETTING))
