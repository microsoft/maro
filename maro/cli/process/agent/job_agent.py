# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import multiprocessing as mp
import os
import subprocess
import time

import psutil
import redis
import yaml


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
            job_detail = json.loads(self.redis_connection.hget("local_process:job_details", pending_job))

            # control process number by parallel
            running_jobs_length = self.redis_connection.hlen("local_process:running_jobs")
            parallel_level = self.redis_connection.hget("local_process:setting", "parallel_level")
            if (
                not parallel_level or
                (running_jobs_length < job_detail["parallel"] and int(parallel_level) > running_jobs_length)
            ):
                self._start_job(job_detail, parallel_level)
                # remove using ticket
                self.redis_connection.lrem("local_process:pending_job_tickets", 0, pending_job)

    def _start_job(self, job_details: dict, parallel_level):
        if (
            not parallel_level or
            job_details["parallel"] < int(parallel_level)
        ):
            self.redis_connection.hset("local_process:setting", "parallel_level", job_details["parallel"])

        pid_list = []
        for component_type, command_info in job_details["components"].items():
            number = command_info["num"]
            command = command_info["command"]
            for num in range(number):
                job_local_path = os.path.expanduser(f"~/.maro/local/{job_details['name']}")
                if not os.path.exists(job_local_path):
                    os.makedirs(job_local_path)

                with open(f"{job_local_path}/{component_type}_{num}.log", "w") as log_file:
                    proc = subprocess.Popen(command, shell=True, stdout=log_file)
                    pid_list.append(proc.pid)

        self.redis_connection.hset("local_process:running_jobs", job_details["name"], json.dumps(pid_list))
        self.redis_connection.hset("local_process:job_status", job_details["name"], json.dumps("running"))


class JobTrackingAgent(mp.Process):
    def __init__(self, redis_connection, check_interval: int = 120):
        super().__init__()
        self.redis_connection = redis_connection
        self.check_interval = check_interval
        self._shutdown_count = 0

    def run(self):
        while True:
            self._check_job_status()
            time.sleep(self.check_interval)
            self._close_process_cli()

    def _check_job_status(self):
        running_jobs = self.redis_connection.hgetall("local_process:running_jobs")

        for running_job, pid_list in running_jobs.items():
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

    def _close_process_cli(self):
        if (
            not self.redis_connection.hlen("local_process:running_jobs") and
            not self.redis_connection.llen("local_process:pending_job_tickets")
        ):
            self._shutdown_count += 1
            print(self._shutdown_count)
        else:
            self._shutdown_count = 0

        if self._shutdown_count >= 5:
            redis_pid = int(self.redis_connection.hget("local_process:setting", "redis_pid"))
            agent_pid = int(self.redis_connection.hget("local_process:setting", "agent_pid"))

            redis_process = psutil.Process(redis_pid)
            redis_process.terminate()

            agent_process = psutil.Process(agent_pid)
            agent_process.terminate()


class KilledJobAgent(mp.Process):
    def __init__(self, redis_connection, check_interval: int = 120):
        super().__init__()
        self.redis_connection = redis_connection
        self.check_interval = check_interval

    def run(self):
        while True:
            self._check_kill_ticket()
            time.sleep(self.check_interval)

    def _check_kill_ticket(self):
        # check pending job ticket
        killed_job_names = self.redis_connection.lrange("local_process:killed_job_tickets", 0, -1)

        for job_name in killed_job_names:
            if self.redis_connection.hexists("local_process:running_jobs", job_name):
                pid_list = json.loads(self.redis_connection.hget("local_process:running_jobs", job_name))
                self._stop_job(job_name, pid_list)
            else:
                self.redis_connection.lrem("local_process:pending_job_tickets", 0, job_name)

            self.redis_connection.lrem("local_process:killed_job_tickets", 0, job_name)

    def _stop_job(self, job_name, pid_list):
        # kill all process by pid
        for pid in pid_list:
            process = psutil.Process(pid)
            process.terminate()

        self.redis_connection.hdel("local_process:running_jobs", job_name)
        self.redis_connection.hset("local_process:job_status", job_name, json.dumps("stop"))


class MasterAgent:
    def __init__(self, check_interval: int = 60):
        with open(os.path.expanduser("~/.maro/local/redis_info.yml"), "r") as rf:
            redis_info = yaml.safe_load(rf)

        self.redis_connection = redis.Redis(host=redis_info["host"], port=redis_info["port"])
        self.check_interval = check_interval

    def start(self) -> None:
        """Start agents."""
        job_tracking_agent = JobTrackingAgent(
            redis_connection=self.redis_connection,
            check_interval=self.check_interval
        )
        job_tracking_agent.start()

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


if __name__ == "__main__":
    master_agent = MasterAgent()
    master_agent.start()
