# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import multiprocessing as mp
import os
import subprocess
import time

import psutil
import redis

from maro.cli.process.utils.details import close_by_pid, get_child_pid, load_setting_info
from maro.cli.utils.params import LocalPaths, ProcessRedisName


class PendingJobAgent(mp.Process):
    def __init__(self, redis_connection, check_interval: int = 60):
        super().__init__()
        self.redis_connection = redis_connection
        self.check_interval = check_interval

    def run(self):
        while True:
            self._check_pending_ticket()
            time.sleep(self.check_interval)

    def _check_pending_ticket(self):
        # Check pending job ticket
        pending_jobs = self.redis_connection.lrange(ProcessRedisName.PENDING_JOB_TICKETS, 0, -1)

        for job_name in pending_jobs:
            job_detail = json.loads(self.redis_connection.hget(ProcessRedisName.JOB_DETAILS, job_name))

            running_jobs_length = self.redis_connection.hlen(ProcessRedisName.RUNNING_JOB)
            parallel_level = self.redis_connection.hget(ProcessRedisName.SETTING, "parallel_level")
            # Start pending job only if current running job's number less than parallel level.
            if int(parallel_level) > running_jobs_length:
                self._start_job(job_detail)
                self.redis_connection.lrem(ProcessRedisName.PENDING_JOB_TICKETS, 0, job_name)

    def _start_job(self, job_details: dict):
        command_pid_list = []
        for component_type, command_info in job_details["components"].items():
            component_number = command_info["num"]
            component_command = command_info["command"]
            for number in range(component_number):
                job_local_path = os.path.expanduser(f"{LocalPaths.MARO_PROCESS}/{job_details['name']}")
                if not os.path.exists(job_local_path):
                    os.makedirs(job_local_path)

                with open(f"{job_local_path}/{component_type}_{number}.log", "w") as log_file:
                    proc = subprocess.Popen(component_command, shell=True, stdout=log_file)
                    command_pid = get_child_pid(proc.pid)
                    command_pid_list.append(command_pid)

        self.redis_connection.hset(ProcessRedisName.RUNNING_JOB, job_details["name"], json.dumps(command_pid_list))


class JobTrackingAgent(mp.Process):
    def __init__(self, redis_connection, check_interval: int = 60):
        super().__init__()
        self.redis_connection = redis_connection
        self.check_interval = check_interval
        self._shutdown_count = 0

    def run(self):
        while True:
            self._check_job_status()
            time.sleep(self.check_interval)
            keep_alive = int(self.redis_connection.hget(ProcessRedisName.SETTING, "keep_agent_alive"))
            if not keep_alive:
                self._close_agents()

    def _check_job_status(self):
        running_jobs = self.redis_connection.hgetall(ProcessRedisName.RUNNING_JOB)
        running_jobs = {job_name.decode(): json.loads(pid_list) for job_name, pid_list in running_jobs}

        for running_job, pid_list in running_jobs.items():
            # Check pid status
            still_alive = False
            for pid in pid_list:
                if psutil.pid_exists(pid):
                    still_alive = True

            # Update if no pid exists
            if not still_alive:
                self.redis_connection.hdel(ProcessRedisName.RUNNING_JOB, running_job)

    def _close_agents(self):
        if (
            not self.redis_connection.hlen(ProcessRedisName.RUNNING_JOB) and
            not self.redis_connection.llen(ProcessRedisName.PENDING_JOB_TICKETS)
        ):
            self._shutdown_count += 1
        else:
            self._shutdown_count = 0

        if self._shutdown_count >= 5:
            agent_pid = int(self.redis_connection.hget(ProcessRedisName.SETTING, "agent_pid"))

            # close agent
            close_by_pid(pid=agent_pid, recursive=True)

            # Set agent status to 0
            self.redis_connection.hset(ProcessRedisName.SETTING, "agent_status", 0)


class KilledJobAgent(mp.Process):
    def __init__(self, redis_connection, check_interval: int = 60):
        super().__init__()
        self.redis_connection = redis_connection
        self.check_interval = check_interval

    def run(self):
        while True:
            self._check_kill_ticket()
            time.sleep(self.check_interval)

    def _check_kill_ticket(self):
        # Check pending job ticket
        killed_job_names = self.redis_connection.lrange(ProcessRedisName.KILLED_JOB_TICKETS, 0, -1)

        for job_name in killed_job_names:
            if self.redis_connection.hexists(ProcessRedisName.RUNNING_JOB, job_name):
                pid_list = json.loads(self.redis_connection.hget(ProcessRedisName.RUNNING_JOB, job_name))
                close_by_pid(pid=pid_list, recursive=False)

                self.redis_connection.hdel(ProcessRedisName.RUNNING_JOB, job_name)
            else:
                self.redis_connection.lrem(ProcessRedisName.PENDING_JOB_TICKETS, 0, job_name)

            self.redis_connection.lrem(ProcessRedisName.KILLED_JOB_TICKETS, 0, job_name)


class MasterAgent:
    def __init__(self):
        setting_info = load_setting_info()
        self.check_interval = setting_info["check_interval"]
        self.redis_connection = redis.Redis(
            host=setting_info["redis_info"]["host"],
            port=setting_info["redis_info"]["port"]
        )
        self.redis_connection.hset(ProcessRedisName.SETTING, "agent_pid", os.getpid())

    def start(self) -> None:
        """Start agents."""
        pending_job_agent = PendingJobAgent(
            redis_connection=self.redis_connection,
            check_interval=self.check_interval
        )
        pending_job_agent.start()

        killed_job_agent = KilledJobAgent(
            redis_connection=self.redis_connection,
            check_interval=self.check_interval
        )
        killed_job_agent.start()

        job_tracking_agent = JobTrackingAgent(
            redis_connection=self.redis_connection,
            check_interval=self.check_interval
        )
        job_tracking_agent.start()


if __name__ == "__main__":
    master_agent = MasterAgent()
    master_agent.start()
