# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import multiprocessing as mp
import os
import psutil
import subprocess
import time

import psutil
import redis

from maro.cli.grass.lib.services.utils.params import JobStatus
from maro.cli.grass.lib.services.utils.subprocess import Subprocess
from maro.cli.process.utils.details import close_by_pid, get_child_pid
from maro.cli.utils.params import LocalPaths, ProcessRedisName
from maro.cli.utils.details_reader import DetailsReader


GET_UTILIZATION_GPUS_COMMAND = "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits"


class PendingJobAgent(mp.Process):
    def __init__(self, cluster_detail: dict, redis_connection, check_interval: int = 60):
        super().__init__()
        self.cluster_detail = cluster_detail
        self.redis_connection = redis_connection
        self.check_interval = check_interval

    def run(self):
        while True:
            self._check_pending_ticket()
            time.sleep(self.check_interval)

    def _check_pending_ticket(self):
        # Check pending job ticket
        pending_jobs = self.redis_connection.lrange(ProcessRedisName.PENDING_JOB_TICKETS, 0, -1)
        running_jobs_length = len(JobTrackingAgent.get_running_jobs(
            self.redis_connection.hgetall(ProcessRedisName.JOB_DETAILS)
        ))
        parallel_level = self.cluster_detail["parallel_level"]

        for job_name in pending_jobs:
            job_detail = json.loads(self.redis_connection.hget(ProcessRedisName.JOB_DETAILS, job_name))
            # Start pending job only if current running job's number less than parallel level.
            if int(parallel_level) > running_jobs_length:
                self._start_job(job_detail)
                self.redis_connection.lrem(ProcessRedisName.PENDING_JOB_TICKETS, 0, job_name)
                running_jobs_length += 1

    def _start_job(self, job_details: dict):
        command_pid_list = []
        for component_type, command_info in job_details["components"].items():
            component_number = command_info["num"]
            component_command = f"JOB_NAME={job_details['name']} " + command_info["command"]
            for number in range(component_number):
                job_local_path = os.path.expanduser(f"{LocalPaths.MARO_PROCESS}/{job_details['name']}")
                if not os.path.exists(job_local_path):
                    os.makedirs(job_local_path)

                with open(f"{job_local_path}/{component_type}_{number}.log", "w") as log_file:
                    proc = subprocess.Popen(component_command, shell=True, stdout=log_file)
                    command_pid = get_child_pid(proc.pid)
                    if not command_pid:
                        command_pid_list.append(proc.pid)
                    else:
                        command_pid_list.append(command_pid)

        job_details["status"] = JobStatus.RUNNING
        job_details["pid_list"] = command_pid_list
        self.redis_connection.hset(ProcessRedisName.JOB_DETAILS, job_details["name"], json.dumps(job_details))


class JobTrackingAgent(mp.Process):
    def __init__(self, cluster_detail: dict, redis_connection, check_interval: int = 60):
        super().__init__()
        self.cluster_detail = cluster_detail
        self.redis_connection = redis_connection
        self.check_interval = check_interval
        self._shutdown_count = 0
        self._countdown = self.redis_connection.hget(ProcessRedisName.SETTING, "agent_countdown")

    def run(self):
        while True:
            self._check_job_status()
            time.sleep(self.check_interval)
            keep_alive = self.cluster_detail["keep_agent_alive"]
            if not keep_alive:
                self._close_agents()

    def _check_job_status(self):
        running_jobs = self.get_running_jobs(self.redis_connection.hgetall(ProcessRedisName.JOB_DETAILS))

        for running_job_name, running_job_detail in running_jobs.items():
            # Check pid status
            still_alive = False
            for pid in running_job_detail["pid_list"]:
                if psutil.pid_exists(pid):
                    still_alive = True

            # Update if no pid exists
            if not still_alive:
                running_job_detail["status"] = JobStatus.FINISH
                del running_job_detail["pid_list"]
                self.redis_connection.hset(
                    ProcessRedisName.JOB_DETAILS,
                    running_job_name,
                    json.dumps(running_job_detail)
                )

    @staticmethod
    def get_running_jobs(job_details: dict):
        running_jobs = {}

        for job_name, job_detail in job_details.items():
            job_detail = json.loads(job_detail)
            if job_detail["status"] == JobStatus.RUNNING:
                running_jobs[job_name.decode()] = job_detail

        return running_jobs

    def _close_agents(self):
        if (
            not len(
                JobTrackingAgent.get_running_jobs(self.redis_connection.hgetall(ProcessRedisName.JOB_DETAILS)
            )) and
            not self.redis_connection.llen(ProcessRedisName.PENDING_JOB_TICKETS)
        ):
            self._shutdown_count += 1
        else:
            self._shutdown_count = 0

        if self._shutdown_count >= self._countdown:
            agent_pid = int(self.redis_connection.hget(ProcessRedisName.SETTING, "agent_pid"))

            # close agent
            close_by_pid(pid=agent_pid, recursive=True)

            # Set agent status to 0
            self.redis_connection.hset(ProcessRedisName.SETTING, "agent_status", 0)


class KilledJobAgent(mp.Process):
    def __init__(self, cluster_detail: dict, redis_connection, check_interval: int = 60):
        super().__init__()
        self.cluster_detail = cluster_detail
        self.redis_connection = redis_connection
        self.check_interval = check_interval

    def run(self):
        while True:
            self._check_killed_tickets()
            time.sleep(self.check_interval)

    def _check_killed_tickets(self):
        # Check pending job ticket
        killed_job_names = self.redis_connection.lrange(ProcessRedisName.KILLED_JOB_TICKETS, 0, -1)

        for job_name in killed_job_names:
            job_detail = json.loads(self.redis_connection.hget(ProcessRedisName.JOB_DETAILS, job_name))
            if job_detail["status"] == JobStatus.RUNNING:
                close_by_pid(pid=job_detail["pid_list"], recursive=False)
                del job_detail["pid_list"]
            elif job_detail["status"] == JobStatus.PENDING:
                self.redis_connection.lrem(ProcessRedisName.PENDING_JOB_TICKETS, 0, job_name)
            elif job_detail["status"] == JobStatus.FINISH:
                continue

            job_detail["status"] = JobStatus.KILLED
            self.redis_connection.hset(ProcessRedisName.JOB_DETAILS, job_name, json.dumps(job_detail))
            self.redis_connection.lrem(ProcessRedisName.KILLED_JOB_TICKETS, 0, job_name)


class MasterAgent:
    def __init__(self):
        self.cluster_detail = DetailsReader.load_cluster_details("process")
        self.check_interval = self.cluster_detail["check_interval"]
        self.redis_connection = redis.Redis(
            host=self.cluster_detail["redis_info"]["host"],
            port=self.cluster_detail["redis_info"]["port"]
        )
        self.redis_connection.hset(ProcessRedisName.SETTING, "agent_pid", os.getpid())

    def start(self) -> None:
        """Start agents."""
        pending_job_agent = PendingJobAgent(
            cluster_detail=self.cluster_detail,
            redis_connection=self.redis_connection,
            check_interval=self.check_interval
        )
        pending_job_agent.start()

        killed_job_agent = KilledJobAgent(
            cluster_detail=self.cluster_detail,
            redis_connection=self.redis_connection,
            check_interval=self.check_interval
        )
        killed_job_agent.start()

        job_tracking_agent = JobTrackingAgent(
            cluster_detail=self.cluster_detail,
            redis_connection=self.redis_connection,
            check_interval=self.check_interval
        )
        job_tracking_agent.start()


if __name__ == "__main__":
    master_agent = MasterAgent()
    master_agent.start()
