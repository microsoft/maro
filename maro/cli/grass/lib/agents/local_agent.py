# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import multiprocessing as mp
import os
import subprocess
import time

import psutil
import redis

from maro.cli.utils.cmp import resource_op
from maro.cli.utils.details import load_cluster_details
from maro.cli.utils.naming import generate_name_with_uuid
from maro.cli.utils.params import LocalPaths, GrassLocalRedisName


START_CONTAINER_COMMAND = (
    "docker run "
    "-it -d "
    "--cpus {cpu} "
    "-m {memory} "
    "--name {container_name} "
    "--network host "
    "{environment_parameters} {labels} "
    "{image_name} {command}"
)


class PendingJobAgent(mp.Process):
    def __init__(self, cluster_name: str, redis_connection, check_interval: int = 60):
        super().__init__()
        self.cluster_name = cluster_name
        self.redis_connection = redis_connection
        self.check_interval = check_interval

    def run(self):
        while True:
            self._check_pending_ticket()
            time.sleep(self.check_interval)

    def _check_pending_ticket(self):
        # Check pending job ticket
        pending_jobs = self.redis_connection.lrange(f"{self.cluster_name}:pending_job_tickets", 0, -1)

        for job_name in pending_jobs:
            job_detail = json.loads(self.redis_connection.hget(f"{self.cluster_name}:job_details", job_name))

            # Allocation
            cluster_resource = json.loads(
                self.redis_connection.hget(GrassLocalRedisName.CLUSTER_DETAILS, self.cluster_name)
            )
            is_satisfied, updated_resource = resource_op(cluster_resource, job_detail["total_request_resource"], "allocate")
            if not is_satisfied:
                continue

            # Start job
            self._start_job(job_detail)
            self.redis_connection.lrem(f"{self.cluster_name}:pending_job_tickets", 0, job_name)

    def _start_job(self, job_details: dict):
        container_name_list = []
        for component_type, command_info in job_details["components"].items():
            for number in range(command_info["num"]):
                container_name = generate_name_with_uuid(prefix=component_type)
                component_command = START_CONTAINER_COMMAND.format(
                    cpu=command_info["resource"]["cpu"],
                    memory=command_info["resource"]["memory"],
                    gpu=command_info["resource"]["gpu"],
                    container_name=container_name,
                    environment_parameters=environment_parameters,
                    labels=labels,
                    image_name=command_info["image"],
                    command=command_info["command"]
                )
            
                completed_process = subprocess.run(
                    component_command,
                    shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf8"
                )
                if completed_process.returncode != 0:
                    raise AllocationFailed(completed_process.stderr)
                container_name_list.append(container_name)

        self.redis_connection.hset(
            f"{self.cluster_name}:running_jobs",
            job_details["name"], # job name
            json.dumps(container_name_list) # container name list
        )


class ContainerTrackingAgent(mp.Process):
    def __init__(self, cluster_name: str, redis_connection, check_interval: int = 60):
        super().__init__()
        self.cluster_name = cluster_name
        self.redis_connection = redis_connection
        self.check_interval = check_interval

    def run(self):
        while True:
            self._check_container_status()
            time.sleep(self.check_interval)

    def _check_container_status(self):
        running_containers = self.redis_connection.hgetall(f"{self.cluster_name}:running_jobs")
        running_containers = {
            job_name.decode(): json.loads(container_name_list) 
            for job_name, container_name_list in running_containers.items()
        }

        for job_name, container_name_list in running_containers.items():
            alive_containers = []
            for container_name in container_name_list:
                # Check container status
                command = f"docker inspect {container_name}"
                completed_process = subprocess.run(
                    command,
                    shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf8"
                )
                return_str = completed_process.stdout
                inspect_details_list = json.loads(return_str)

                # Update container status
                if not inspect_details_list[0]["State"]["Running"]:
                    self._container_exit(job_name, container_name, inspect_details_list[0])
                else:
                    alive_containers.append(container_name)

            self.redis_connection.hset(f"{self.cluster_name}:running_jobs", job_name, json.dumps(alive_containers))

    def _container_exit(self, job_name: str, container_name: str, inspect_details: dict):
        if inspect_details["State"]["ExitCode"] != 0:
            # Unsuccessfully exited
            job_details = json.loads(self.redis_connection.hget(f"{self.cluster_name}:job_details", job_name))
            job_details["State"] = "Failed"
            job_details["ExitCode"] = inspect_details["State"]["ExitCode"]
            job_details["Error"] = inspect_details["State"]["Error"]
            self.redis_connection.hset(f"{self.cluster_name}:job_details", job_name, json.dumps(job_details))


class JobTrackingAgent(mp.Process):
    def __init__(self, cluster_name: str, redis_connection, check_interval: int = 60):
        super().__init__()
        self.cluster_name = cluster_name
        self.redis_connection = redis_connection
        self.check_interval = check_interval

    def run(self):
        while True:
            self._check_job_state()
            time.sleep(self.check_interval)

    def _check_job_state(self):
        running_containers = self.redis_connection.hgetall(f"{self.cluster_name}:running_jobs")
        running_containers = {
            job_name.decode(): json.loads(container_name_list) 
            for job_name, container_name_list in running_containers.items()
        }

        for job_name, container_list in running_jobs.items():
            job_details = json.loads(
                self.redis_connection.hget(f"{self.cluster_name}:job_details", job_name)
            )

            if job_detail["State"] == "Failed":
                self._stop_containers(container_list)
                self.redis_connection.hdel(f"{self.cluster_name}:running_jobs", job_name)
                self._job_clear(job_name, job_details["total_request_resource"])
                continue
            
            if not container_list:
                job_details["State"] == "Finish"
                self.redis_connection.hset(f"{self.cluster_name}:job_details", job_name, json.loads(job_details))
                self._job_clear(job_name, job_details["total_request_resource"])
    
    def _stop_containers(self, container_list: list):
        for container_name in container_list:
            command = f"docker stop {container_name}"
            completed_process = subprocess.run(
                command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf8"
            )
            if completed_process.returncode != 0:
                raise AllocationFailed(completed_process.stderr)

    def _job_clear(self, job_name: str, release_resource: dict):
        cluster_resource = json.loads(
            self.redis_connection.hget(GrassLocalRedisName.CLUSTER_DETAILS, self.cluster_name)
        )

        # resource release
        _, updated_resource = resource_op(cluster_resource, release_resource, op="release")

        self.redis_connection.hdel(f"{self.cluster_name}:running_jobs", job_name)


class KilledJobAgent(mp.Process):
    def __init__(self, cluster_name: str, redis_connection, check_interval: int = 60):
        super().__init__()
        self.cluster_name = cluster_name
        self.redis_connection = redis_connection
        self.check_interval = check_interval

    def run(self):
        while True:
            self._check_kill_ticket()
            time.sleep(self.check_interval)

    def _check_kill_ticket(self):
        # Check pending job ticket
        killed_job_names = self.redis_connection.lrange(f"{self.cluster_name}:killed_job_tickets", 0, -1)

        for job_name in killed_job_names:
            if self.redis_connection.hexists(f"{self.cluster_name}:running_jobs", job_name):
                self._kill_job(job_name)
            else:
                self.redis_connection.lrem(f"{self.cluster_name}:pending_job_tickets", 0, job_name)

            self.redis_connection.lrem(f"{self.cluster_name}:killed_job_tickets", 0, job_name)
    
    def _kill_job(self, job_name):
        # Stop all related containers
        container_list = json.loads(self.redis_connection.hget(f"{self.cluster_name}:running_jobs", job_name))
        for container_name in container_list:
            command = f"docker stop {container_name}"
            completed_process = subprocess.run(
                command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf8"
            )
            if completed_process.returncode != 0:
                raise AllocationFailed(completed_process.stderr)
        
        # Update job state
        job_detail = json.loads(self.redis_connection.hget(f"{self.cluster_name}:job_details", job_name))
        job_detail["State"] = "Killed"
        self.redis_connection.hset(f"{self.cluster_name}:job_details", job_name, json.dumps(job_detail))
        self.redis_connection.hdel(f"{self.cluster_name}:running_jobs", job_name)

        # Release resource
        cluster_resource = json.loads(
            self.redis_connection.hget(GrassLocalRedisName.CLUSTER_DETAILS, self.cluster_name)
        )
        job_resource = job_detail["total_request_resource"]
        _, updated_resource = resource_op(cluster_resource, job_resource, op="release")



class MasterAgent:
    def __init__(self, cluster_name: str):
        self.cluster_detail = load_cluster_details(cluster_name)
        self.check_interval = self.cluster_detail["agents"]["check_interval"]
        self.redis_connection = redis.Redis(
            host="localhost",
            port=self.cluster_detail["redis"]["port"]
        )

    def start(self) -> None:
        """Start agents."""
        pending_job_agent = PendingJobAgent(
            cluster_name=self.cluster_detail["name"],
            redis_connection=self.redis_connection,
            check_interval=self.check_interval
        )
        pending_job_agent.start()

        killed_job_agent = KilledJobAgent(
            cluster_name=self.cluster_detail["name"],
            redis_connection=self.redis_connection,
            check_interval=self.check_interval
        )
        killed_job_agent.start()

        job_tracking_agent = JobTrackingAgent(
            cluster_name=self.cluster_detail["name"],
            redis_connection=self.redis_connection,
            check_interval=self.check_interval
        )
        job_tracking_agent.start()

        container_tracking_agent = ContainerTrackingAgent(
            cluster_name=self.cluster_detail["name"],
            redis_connection=self.redis_connection,
            check_interval=self.check_interval
        )
        container_tracking_agent.start()


if __name__ == "__main__":
    master_agent = MasterAgent()
    master_agent.start()
