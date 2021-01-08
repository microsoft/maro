# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import json
import os
import psutil
import shutil
import subprocess

import redis

from maro.cli.utils.cmp import resource_op
from maro.cli.utils.details import (
    load_cluster_details, load_job_details, load_schedule_details, 
    save_cluster_details, save_job_details, save_schedule_details
)
from maro.cli.utils.params import GlobalPaths, GrassLocalRedisName, LocalPaths
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)


class GrassLocalExecutor:

    def __init__(self, cluster_name: str):
        self.cluster_name = cluster_name
        self.cluster_details = load_cluster_details(cluster_name=cluster_name)
        
        # Connection with Redis
        redis_port = self.cluster_details["master"]["redis"]["port"]
        try:
            self._redis_connection = redis.Redis(host="localhost", port=redis_port)
            self._redis_connection.ping()
        except:
            redis_process = subprocess.Popen(
                ["redis-server", "--port", str(redis_port), "--daemonize yes"]
            )
            redis_process.wait(timeout=2)

    @staticmethod
    def build_cluster_details(create_deployment: dict):
        # Get cluster name and save details
        cluster_name = create_deployment["name"]
        if os.path.isdir(f"{GlobalPaths.ABS_MARO_CLUSTERS}/{cluster_name}"):
            raise BadRequestError(f"Cluster '{cluster_name}' is exist.")
        os.makedirs(f"{GlobalPaths.ABS_MARO_CLUSTERS}/{cluster_name}")
        save_cluster_details(
            cluster_name=cluster_name,
            cluster_details=create_deployment,
            sync=False
        )

    def _standardize_local_deployment(self, deployment: dict):
        total_cpu, total_memory, total_gpu = 0, 0, 0
        for component_type, component_dict in deployment["components"]:
            total_cpu += component_dict["num"] * component_dict["resource"]["cpu"]
            total_memory += component_dict["num"] * component_dict["resource"]["memory"]
            total_gpu += component_dict["num"] * component_dict["resource"]["gpu"]
        deployment["total_request_resource"] = {
            "cpu": total_cpu,
            "memory": total_memory,
            "gpu": total_gpu
        }
        
        return deployment

    def create(self):
        logger.info("Creating cluster")

        # Allocation
        cluster_resource = self.cluster_details["resource"]
        if self._redis_connection.exists(GrassLocalRedisName.RUNTIME_DETAILS):
            available_resource = json.loads(
                self._redis_connection.hget(GrassLocalRedisName.RUNTIME_DETAILS, "available_resource")
            )
        else:
            cpu_count = psutil.cpu_count(logical=False)
            memory_info = psutil.virtual_memory()
            total_memory = round(float(memory_info.total)/(1024**3), 2)
            free_memory = round(float(memory_info.free)/(1024**3), 2)
            # Add GPU later
            available_resource = {"cpu": cpu_count,
                                  "memory": free_memory,
                                  "gpu": 0}
        
        # Update resource
        is_satisfied, updated_resource = resource_op(available_resource, cluster_resource, op="allocate")
        if not is_satisfied:
            shutil.rmtree(f"{GlobalPaths.ABS_MARO_CLUSTERS}/{self.cluster_name}", True)
            raise BadRequestError("No enough resource for this cluster.")

        self._redis_connection.hset(
            GrassLocalRedisName.RUNTIME_DETAILS, "available_resource", json.dumps(updated_resource)
        )

        # Push cluster details into Redis
        self._redis_connection.hset(
            GrassLocalRedisName.CLUSTER_DETAILS, 
            self.cluster_name, 
            json.dumps(self.cluster_details["resource"])
        )

    def delete(self):
        # Get Redis resource
        try:
            available_resource = json.loads(
                self._redis_connection.hget(GrassLocalRedisName.RUNTIME_DETAILS, "available_resource")
            )
        except:
            raise BadRequestError(f"Failure to get runtime details from Redis. {self.cluster_name} cannot delete.")

        cluster_resource = self.cluster_details["resource"]

        # Update resource
        _, updated_resource = resource_op(available_resource, cluster_resource, "release")

        self._redis_connection.hset(GrassLocalRedisName.RUNTIME_DETAILS, "available_resource", json.loads(updated_resource))
        self._redis_connection.hdel(GrassLocalRedisName.CLUSTER_DETAILS, self.cluster_name)
        
        shutil.rmtree(f"{GlobalPaths.ABS_MARO_CLUSTERS}/{self.cluster_name}", True)

        
    def start_job(self, deployment_path: str):
        # Load start_job_deployment
        with open(deployment_path, "r") as fr:
            start_job_deployment = yaml.safe_load(fr)
        
        start_job_deployment = self._standardize_local_deployment(start_job_deployment)

        # Check resource
        is_satisfied, _ = resource_op(self.cluster_details["resource"], start_job_deployment["total_request_resource"], op="allocate")
        if not is_satisfied:
            raise BadRequestError(f"No enough resource to start job {job_name} in {self.cluster_name}.")

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
        try:
            job_details = json.loads(
                self._redis_connection.hget(f"{self.cluster_name}:job_details", job_name)
            )
        except:
            logger.error(f"No such job '{job_name}' in {self.cluster_name}.")
            return

        # push job_name into killed_job_tickets
        self._redis_connection.lpush(
            f"{self.cluster_name}:killed_job_ticket",
            job_name
        )
        logger.info(f"Sending {job_name} into killed job tickets.")

    def get_job_logs(self, job_name: str):
        source_path = os.path.expanduser(f"{GlobalPaths.ABS_MARO_CLUSTERS}/{self.cluster_name}/{job_name}")
        if not os.path.exists(source_path):
            logger.error(f"Cannot find the logs of {job_name}.")

        destination = os.path.join(os.getcwd(), job_name)
        if os.path.exists(destination):
            shutil.rmtree(destination)
        shutil.copytree(source_path, destination)
        logger.info(f"Dump logs in path: {destination}.")

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
        start_schedule_deployment = self._standardize_local_deployment(start_schedule_deployment)
        
        # Check resource
        is_satisfied, _ = resource_op(self.cluster_details["resource"], start_schedule_deployment["total_request_resource"], op="allocate")
        if not is_satisfied:
            raise BadRequestError(f"No enough resource to start schedule {schedule_name} in {self.cluster_name}.")

        # Save schedule deployment
        save_schedule_details(
            cluster_name=self.cluster_name,
            schedule_name=schedule_name,
            schedule_details=start_schedule_deployment
        )
        
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
            job_detail["cluster_name"] = self.cluster_name

            self._push_pending_job(job_detail)

    def stop_schedule(self, schedule_name: str):
        try:
            schedule_details = json.loads(
                self._redis_connection.hget(f"{self.cluster_name}:job_details", schedule_name)
            )
        except:
            logger.error(f"No such schedule '{schedule_name}' in Redis.")
            return

        job_list = schedule_details["job_names"]

        for job_name in job_list:
            self.stop_job(job_name)
