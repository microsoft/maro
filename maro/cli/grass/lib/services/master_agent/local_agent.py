# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import multiprocessing as mp
import os
import subprocess
import sys
import time

import redis

from maro.cli.grass.lib.services.utils.exception import ResourceAllocationFailed
from maro.cli.grass.lib.services.utils.name_creator import NameCreator
from maro.cli.grass.lib.services.utils.params import JobStatus
from maro.cli.utils.cmp import ResourceOperation, resource_op
from maro.cli.utils.details_reader import DetailsReader

START_CONTAINER_COMMAND = (
    "sudo docker run "
    "-it -d "
    "--cpus {cpu} "
    "-m {memory} "
    "--name {container_name} "
    "--network host "
    "-v {volumes} "
    "{environment_parameters} {labels} "
    "{image_name} {command}"
)

START_CONTAINER_WITH_GPU_COMMAND = (
    "sudo docker run "
    "-it -d "
    "--cpus {cpu} "
    "-m {memory} "
    "--gpus {gpu} "
    "--name {container_name} "
    "--network host "
    "-v {volumes} "
    "{environment_parameters} {labels} "
    "{image_name} {command}"
)

ERROR_CODES_FOR_NOT_RESTART_CONTAINER = {0, 64, 65}

UNFINISHED_JOB_STATUS = [JobStatus.PENDING, JobStatus.RUNNING]


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
                self.redis_connection.hget(f"{self.cluster_name}:runtime_detail", "available_resource")
            )
            is_satisfied, updated_resource = resource_op(
                cluster_resource,
                job_detail["total_request_resource"],
                ResourceOperation.ALLOCATION
            )
            if not is_satisfied:
                continue

            # Start job
            self._start_job(job_detail)
            self.redis_connection.lrem(f"{self.cluster_name}:pending_job_tickets", 0, job_name)
            # Update resource
            cluster_resource = json.loads(
                self.redis_connection.hget(f"{self.cluster_name}:runtime_detail", "available_resource")
            )
            is_satisfied, updated_resource = resource_op(
                cluster_resource,
                job_detail["total_request_resource"],
                ResourceOperation.ALLOCATION
            )
            self.redis_connection.hset(
                f"{self.cluster_name}:runtime_detail",
                "available_resource",
                json.dumps(updated_resource)
            )

    def _start_job(self, job_detail: dict):
        container_name_list = []
        for component_type, command_info in job_detail["components"].items():
            for number in range(command_info["num"]):
                container_name = NameCreator.create_name_with_uuid(prefix=component_type)
                environment_parameters = (
                    f"-e CONTAINER_NAME={container_name} "
                    f"-e JOB_NAME={job_detail['name']} "
                )
                labels = (
                    f"-l CONTAINER_NAME={container_name} "
                    f"-l JOB_NAME={job_detail['name']} "
                )
                if int(command_info["resources"]["gpu"]) == 0:
                    component_command = START_CONTAINER_COMMAND.format(
                        cpu=command_info["resources"]["cpu"],
                        memory=command_info["resources"]["memory"],
                        container_name=container_name,
                        volumes=command_info["mount"]["target"],
                        environment_parameters=environment_parameters,
                        labels=labels,
                        image_name=command_info["image"],
                        command=command_info["command"]
                    )
                else:
                    component_command = START_CONTAINER_WITH_GPU_COMMAND.format(
                        cpu=command_info["resources"]["cpu"],
                        memory=command_info["resources"]["memory"],
                        gpu=command_info["resources"]["gpu"],
                        container_name=container_name,
                        volumes=command_info["mount"]["target"],
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
                    raise ResourceAllocationFailed(completed_process.stderr)
                container_name_list.append(container_name)

        job_detail["status"] = JobStatus.RUNNING
        job_detail["container_name_list"] = container_name_list
        self.redis_connection.hset(
            f"{self.cluster_name}:job_details",
            job_detail["name"],
            json.dumps(job_detail)
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
        running_jobs = ContainerTrackingAgent.get_running_jobs(
            self.redis_connection.hgetall(f"{self.cluster_name}:job_details")
        )

        for job_name, job_detail in running_jobs.items():
            alive_containers = []
            container_name_list = job_detail["container_name_list"]
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
                    is_continuous = self._container_exit(job_detail, inspect_details_list[0])
                    if not is_continuous:
                        break
                else:
                    alive_containers.append(container_name)

            if not alive_containers and is_continuous:
                job_detail["status"] = JobStatus.FINISH
                self.redis_connection.hset(f"{self.cluster_name}:job_details", job_name, json.dumps(job_detail))

    @staticmethod
    def get_running_jobs(job_details: dict):
        name_to_running_job_detail = {}
        for job_name, job_detail in job_details.items():
            job_detail = json.loads(job_detail)
            if job_detail["status"] == JobStatus.RUNNING:
                name_to_running_job_detail[job_name.decode()] = job_detail

        return name_to_running_job_detail

    def _container_exit(self, job_detail: dict, inspect_details: dict):
        exit_code = int(inspect_details["State"]["ExitCode"])
        if exit_code not in ERROR_CODES_FOR_NOT_RESTART_CONTAINER:
            # Unsuccessfully exited
            job_detail["status"] = JobStatus.FAILED
            job_detail["ExitCode"] = exit_code
            job_detail["Error"] = inspect_details["State"]["Error"]
            self.redis_connection.hset(f"{self.cluster_name}:job_details", job_detail["name"], json.dumps(job_detail))

            return False

        return True


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
        finished_jobs = self._get_finished_jobs(
            self.redis_connection.hgetall(f"{self.cluster_name}:job_details")
        )

        for job_name, job_detail in finished_jobs.items():
            if job_detail["status"] in [JobStatus.KILLED, JobStatus.FAILED]:
                self._stop_containers(job_detail["container_name_list"])

            self._job_clear(job_name, job_detail["total_request_resource"])

            self.redis_connection.hset(f"{self.cluster_name}:job_details", job_name, json.dumps(job_detail))

    def _get_finished_jobs(self, job_details: dict):
        finished_jobs = {}
        for job_name, job_detail in job_details.items():
            job_detail = json.loads(job_detail)
            if "checked" not in job_detail and job_detail["status"] not in UNFINISHED_JOB_STATUS:
                job_detail["checked"] = 1
                finished_jobs[job_name.decode()] = job_detail

        return finished_jobs

    def _stop_containers(self, container_list: list):
        for container_name in container_list:
            command = f"docker stop {container_name}"
            completed_process = subprocess.run(
                command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf8"
            )
            if completed_process.returncode != 0:
                raise ResourceAllocationFailed(completed_process.stderr)

    def _job_clear(self, job_name: str, release_resource: dict):
        cluster_resource = json.loads(
            self.redis_connection.hget(f"{self.cluster_name}:runtime_detail", "available_resource")
        )

        # resource release
        _, updated_resource = resource_op(
            cluster_resource, release_resource, ResourceOperation.RELEASE
        )

        self.redis_connection.hset(
            f"{self.cluster_name}:runtime_detail",
            "available_resource",
            json.dumps(updated_resource)
        )


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
            self.redis_connection.lrem(f"{self.cluster_name}:pending_job_tickets", 0, job_name)
            job_detail = json.loads(self.redis_connection.hget(f"{self.cluster_name}:job_details", job_name))
            if job_detail["status"] in UNFINISHED_JOB_STATUS:
                if job_detail["status"] == JobStatus.PENDING:
                    job_detail["checked"] = 1
                job_detail["status"] = JobStatus.KILLED

                self.redis_connection.hset(f"{self.cluster_name}:job_details", job_name, json.dumps(job_detail))

            self.redis_connection.lrem(f"{self.cluster_name}:killed_job_tickets", 0, job_name)


class MasterAgent:
    def __init__(self, cluster_name: str):
        self.cluster_name = cluster_name
        self.cluster_detail = DetailsReader.load_cluster_details(cluster_name)
        self.check_interval = self.cluster_detail["master"]["agents"]["check_interval"]
        self.redis_connection = redis.Redis(
            host="localhost",
            port=self.cluster_detail["master"]["redis"]["port"]
        )
        self.redis_connection.hset(f"{self.cluster_name}:runtime_detail", "agent_id", os.getpid())

    def start(self) -> None:
        """Start agents."""
        pending_job_agent = PendingJobAgent(
            cluster_name=self.cluster_name,
            redis_connection=self.redis_connection,
            check_interval=self.check_interval
        )
        pending_job_agent.start()

        killed_job_agent = KilledJobAgent(
            cluster_name=self.cluster_name,
            redis_connection=self.redis_connection,
            check_interval=self.check_interval
        )
        killed_job_agent.start()

        job_tracking_agent = JobTrackingAgent(
            cluster_name=self.cluster_name,
            redis_connection=self.redis_connection,
            check_interval=self.check_interval
        )
        job_tracking_agent.start()

        container_tracking_agent = ContainerTrackingAgent(
            cluster_name=self.cluster_name,
            redis_connection=self.redis_connection,
            check_interval=self.check_interval
        )
        container_tracking_agent.start()


if __name__ == "__main__":
    cluster_name = sys.argv[1]
    master_agent = MasterAgent(cluster_name)
    master_agent.start()
