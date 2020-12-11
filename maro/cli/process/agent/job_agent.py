# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import psutil
import redis
import time
import subprocess

import multiprocessing as mp

from maro.utils.exception.cli_exception import CliException


class PendingJobAgent(mp.Process):
    def __init__(self, redis_connection, check_interval: int = 120):
        super().__init__()
        self.redis_connection = redis_connection
        self.check_interval = check_interval

    def run(self):
        while True:
            self._check_pending_ticket()
            time.sleep(self.check_interval)

    def _check_pending_ticket(self):
        # check pending job ticket
        pending_jobs = self.redis_connection.lrange("local_process:pending_job_tickets", 0, -1)

        for pending_job in pending_jobs:
            job_details = json.loads(self.redis_connection.hget("local_process:job_details", pending_job))

            # control process number by parallel
            running_jobs_length = self.redis_connection.hlen("local_process:running_jobs")
            parallel_level = self.redis_connection.get("local_process:parallel_level")
            if (
                not self.redis_connection.exists("local_process:parallel_level") or
                (running_jobs_length < job_details["parallel"] and int(parallel_level) > running_jobs_length)
            ):
                self._start_job(job_details)
                # remove using ticket
                self.redis_connection.lrem("local_process:pending_job_tickets", 0, pending_job)

    def _start_job(self, job_details: dict):
        if (
            not self.redis_connection.exists("local_process:parallel_level") or
            job_details["parallel"] < int(self.redis_connection.get("local_process:parallel_level"))
        ):
            self.redis_connection.set("local_process:parallel_level", job_details["parallel"])

        pid_list = []
        for component_type, command_info in job_details["components"]:
            number = command_info["num"]
            command = command_info["command"]
            for num in range(number):
                with open(f"~/.maro/local/{job_details['name']}/{component_type}_{num}.log", "w") as log_file:
                    proc = subprocess.Popen(command, shell=True, stdout=log_file)
                    pid_list.append(proc.pid)

        self.redis_connection.hset("local_process:running_jobs", job_details["name"], json.dumps(pid_list))
        self.redis_connection.hset("local_process:job_status", job_details["name"], json.dumps("running"))


class JobTrackingAgent(mp.Process):
    def __init__(self, redis_connection, check_interval: int = 120):
        super().__init__()
        self.redis_connection = redis_connection
        self.check_interval = check_interval

    def run(self):
        while True:
            self._check_job_status()
            time.sleep(self.check_interval)

    def _check_job_status(self):
        running_jobs = self.redis_connection.hgetall("local_process:running_jobs")

        for running_job, pid_list in running_jobs:
            # check pid status
            pid_list = json.loads(pid_list)
            still_alive = False
            for pid in pid_list:
                if psutil.pid_exists(pid):
                    still_alive = True

            # update if any finished or error
            if not still_alive:
                self.redis_connection.hdel("local_process:running_jobs", running_job)
                self.redis_connection.hset("local_process:job_status", running_job, json.dumps("finish"))


class KilledJobAgent(mp.Process):
    def __init__(self, redis_connection, check_interval: int = 120):
        super().__init__()
        self.redis_connection = redis_connection
        self.check_interval = check_interval

    def run(self):
        while True:
            self._check_job_status()
            time.sleep(self.check_interval)

    def _check_kill_ticket(self):
        # check pending job ticket
        killed_job_names = self.redis_connection.lrange("local_process:kill_job_tickets", 0, -1)

        for job_name in killed_job_names:
            if self.redis_connection.hexists("local_process:running_jobs", job_name):
                pid_list = self.redis_connection.hget("local_process:running_jobs", job_name)
                self._stop_job(job_name, pid_list)
            # else:
            #     raise CliException(f"Unable to stop {job_name}. It is not in running job list.")

            self.redis_connection.lrem("local_process:kill_job_tickets", 0, job_name)

    def _stop_job(self, job_name, pid_list):
        # kill all process by pid
        for pid in pid_list:
            process = psutil.Process(pid)
            process.terminate()

        self.redis_connection.hdel("local_process:running_jobs", job_name)
        self.redis_connection.hset("local_process:job_status", job_name, json.dumps("stop"))


class MasterAgent:
    def __init__(self, redis_info, check_interval: int = 60):
        self.redis_connection = redis.Redis(host=redis_info["host"], port=redis_info["port"])
        self.check_interval = check_interval

    def start(self) -> None:
        """Start agents."""
        job_tracking_agent = JobTrackingAgent(redis_connection=self.redis_connection, check_interval=self.check_interval)
        job_tracking_agent.start()
        pending_job_agent = PendingJobAgent(redis_connection=self.redis_connection, check_interval=self.check_interval)
        pending_job_agent.start()
        killed_job_agent = KilledJobAgent(redis_connection=self.redis_connection, check_interval=self.check_interval)
        killed_job_agent.start()
