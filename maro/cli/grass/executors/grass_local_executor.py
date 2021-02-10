# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import json
import os
import shutil
import subprocess

import redis
import yaml

from maro.cli.grass.lib.services.utils.params import JobStatus
from maro.cli.process.utils.details import close_by_pid
from maro.cli.utils.abs_visible_executor import AbsVisibleExecutor
from maro.cli.utils.cmp import ResourceOperation, resource_op
from maro.cli.utils.details_reader import DetailsReader
from maro.cli.utils.details_writer import DetailsWriter
from maro.cli.utils.params import GlobalPaths, LocalPaths
from maro.cli.utils.resource_executor import LocalResourceExecutor
from maro.utils.exception.cli_exception import BadRequestError
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)


class GrassLocalExecutor(AbsVisibleExecutor):
    def __init__(self, cluster_name: str, cluster_details: dict = None):
        self.cluster_name = cluster_name
        self.cluster_details = DetailsReader.load_cluster_details(cluster_name=cluster_name) \
            if not cluster_details else cluster_details

        # Connection with Redis
        redis_port = self.cluster_details["master"]["redis"]["port"]
        self._redis_connection = redis.Redis(host="localhost", port=redis_port)
        try:
            self._redis_connection.ping()
        except Exception:
            redis_process = subprocess.Popen(
                ["redis-server", "--port", str(redis_port), "--daemonize yes"]
            )
            redis_process.wait(timeout=2)

        # Connection with Resource Redis
        self._resource_redis = LocalResourceExecutor()

    def _completed_local_job_deployment(self, deployment: dict):
        total_cpu, total_memory, total_gpu = 0, 0, 0
        for component_type, component_dict in deployment["components"].items():
            total_cpu += int(component_dict["num"]) * int(component_dict["resources"]["cpu"])
            total_memory += int(component_dict["num"]) * int(component_dict["resources"]["memory"][:-1])
            total_gpu += int(component_dict["num"]) * int(component_dict["resources"]["gpu"])
        deployment["total_request_resource"] = {
            "cpu": total_cpu,
            "memory": total_memory,
            "gpu": total_gpu
        }

        deployment["status"] = JobStatus.PENDING

        return deployment

    def create(self):
        logger.info("Creating cluster")

        # Get cluster name and save cluster details.
        if os.path.isdir(f"{GlobalPaths.ABS_MARO_CLUSTERS}/{self.cluster_name}"):
            raise BadRequestError(f"Cluster '{self.cluster_name}' is exist.")

        # Build connection with Resource Redis
        self._resource_redis.add_cluster()

        # Allocation
        cluster_resource = self.cluster_details["master"]["resource"]
        available_resource = self._resource_redis.get_available_resource()

        # Update resource
        is_satisfied, updated_resource = resource_op(
            available_resource, cluster_resource, ResourceOperation.ALLOCATION
        )
        if not is_satisfied:
            self._resource_redis.sub_cluster()
            raise BadRequestError("No enough resource for this cluster.")

        self._resource_redis.set_available_resource(updated_resource)

        # Start agents.
        self._agents_start()

        # Set available resource for cluster
        self._redis_connection.hset(
            f"{self.cluster_name}:runtime_detail",
            "available_resource",
            json.dumps(cluster_resource)
        )

        # Save cluster config locally.
        DetailsWriter.save_cluster_details(
            cluster_name=self.cluster_name,
            cluster_details=self.cluster_details
        )

        logger.info(f"{self.cluster_name} is created.")

    def delete(self):
        logger.info(f"Deleting cluster {self.cluster_name}")

        # Remove local cluster file.
        shutil.rmtree(f"{GlobalPaths.ABS_MARO_CLUSTERS}/{self.cluster_name}", True)

        # Stop cluster agents.
        self._agents_stop()

        # Release cluster resource.
        available_resource = self._resource_redis.get_available_resource()

        # Update resource
        cluster_resource = self.cluster_details["master"]["resource"]
        _, updated_resource = resource_op(
            available_resource, cluster_resource, ResourceOperation.RELEASE
        )
        self._resource_redis.set_available_resource(updated_resource)

        # Rm connection from resource redis.
        self._resource_redis.sub_cluster()

        # Clear local redis data.
        self._redis_clear()

        logger.info(f"{self.cluster_name} is deleted.")

    def _agents_start(self):
        command = f"python {LocalPaths.MARO_GRASS_LOCAL_AGENT} {self.cluster_name}"
        _ = subprocess.Popen(command, shell=True)

    def _agents_stop(self):
        try:
            agent_pid = int(self._redis_connection.hget(f"{self.cluster_name}:runtime_detail", "agent_id"))
            close_by_pid(agent_pid, recursive=True)
        except Exception as e:
            logger.warning(f"Failure to close {self.cluster_name}'s agents, due to {e}")

    def _redis_clear(self):
        redis_keys = self._redis_connection.keys(f"{self.cluster_name}:*")
        for key in redis_keys:
            self._redis_connection.delete(key)

    def start_job(self, deployment_path: str):
        # Load start_job_deployment
        with open(deployment_path, "r") as fr:
            start_job_deployment = yaml.safe_load(fr)

        start_job_deployment = self._completed_local_job_deployment(start_job_deployment)

        # Check resource
        is_satisfied, _ = resource_op(
            self.cluster_details["master"]["resource"],
            start_job_deployment["total_request_resource"],
            ResourceOperation.ALLOCATION
        )
        if not is_satisfied:
            raise BadRequestError(f"No enough resource to start job {start_job_deployment['name']}.")

        self._push_pending_job(start_job_deployment)

    def _push_pending_job(self, job_details: dict):
        # Check cluster has enough resource
        job_name = job_details["name"]

        # Push job details to redis
        self._redis_connection.hset(
            f"{self.cluster_name}:job_details",
            job_name,
            json.dumps(job_details)
        )

        # Push job name to pending_job_tickets
        self._redis_connection.lpush(
            f"{self.cluster_name}:pending_job_tickets",
            job_name
        )
        logger.info(f"Sending {job_name} into pending job tickets.")

    def stop_job(self, job_name: str):
        if not self._redis_connection.hexists(f"{self.cluster_name}:job_details", job_name):
            logger.error(f"No such job '{job_name}' in {self.cluster_name}.")
            return

        # push job_name into killed_job_tickets
        self._redis_connection.lpush(
            f"{self.cluster_name}:killed_job_tickets",
            job_name
        )
        logger.info(f"Sending {job_name} into killed job tickets.")

    def get_job_logs(self, job_name: str):
        job_detail = json.loads(self._redis_connection.hget(f"{self.cluster_name}:job_details", job_name))

        try:
            container_ids = job_detail["container_name_list"]
        except Exception:
            logger.warning(f"{job_name} is not started.")
            return

        destination = os.path.join(os.getcwd(), job_name)
        if not os.path.exists(destination):
            os.mkdir(destination)

        for container_id in container_ids:
            command = f"docker logs {container_id}"
            with open(f"{destination}/{container_id}.log", "w") as log_file:
                _ = subprocess.Popen(command, shell=True, stdout=log_file)

    def list_job(self):
        # Get all jobs
        jobs = self._redis_connection.hgetall(f"{self.cluster_name}:job_details")
        for job_name, job_details in jobs.items():
            job_details = json.loads(job_details)

            logger.info(job_details)

    def start_schedule(self, deployment_path: str):
        # Load start_schedule_deployment
        with open(deployment_path, "r") as fr:
            start_schedule_deployment = yaml.safe_load(fr)

        schedule_name = start_schedule_deployment["name"]
        start_schedule_deployment = self._completed_local_job_deployment(start_schedule_deployment)

        # Check resource
        is_satisfied, _ = resource_op(
            self.cluster_details["master"]["resource"],
            start_schedule_deployment["total_request_resource"],
            ResourceOperation.ALLOCATION
        )
        if not is_satisfied:
            raise BadRequestError(f"No enough resource to start schedule {schedule_name} in {self.cluster_name}.")

        # push schedule details to Redis
        self._redis_connection.hset(
            f"{self.cluster_name}:job_details",
            schedule_name,
            json.dumps(start_schedule_deployment)
        )

        job_list = start_schedule_deployment["job_names"]
        # switch schedule details into job details
        job_detail = copy.deepcopy(start_schedule_deployment)
        del job_detail["job_names"]

        for job_name in job_list:
            job_detail["name"] = job_name

            self._push_pending_job(job_detail)

    def stop_schedule(self, schedule_name: str):
        try:
            schedule_details = json.loads(
                self._redis_connection.hget(f"{self.cluster_name}:job_details", schedule_name)
            )
        except Exception:
            logger.error(f"No such schedule '{schedule_name}' in Redis.")
            return

        if "job_names" not in schedule_details:
            logger.error(f"'{schedule_name}' is not a schedule.")
            return

        job_list = schedule_details["job_names"]

        for job_name in job_list:
            self.stop_job(job_name)

    def get_job_details(self):
        jobs = self._redis_connection.hgetall(f"{self.cluster_name}:job_details")
        for job_name, job_details_str in jobs.items():
            jobs[job_name] = json.loads(job_details_str)

        return list(jobs.values())

    def get_job_queue(self):
        pending_job_queue = self._redis_connection.lrange(
            f"{self.cluster_name}:pending_job_tickets",
            0, -1
        )
        killed_job_queue = self._redis_connection.lrange(
            f"{self.cluster_name}:killed_job_tickets",
            0, -1
        )
        return {
            "pending_jobs": pending_job_queue,
            "killed_jobs": killed_job_queue
        }

    def get_resource(self):
        return self.cluster_details["master"]["resource"]

    def get_resource_usage(self, previous_length: int = 0):
        available_resource = self._redis_connection.hget(
            f"{self.cluster_name}:runtime_detail",
            "available_resource"
        )
        return json.loads(available_resource)
