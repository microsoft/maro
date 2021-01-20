# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import json
import os
import shutil

from maro.cli.process.utils.details import env_prepare, load_details
from maro.cli.utils.params import LocalPaths, ProcessRedisName
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)


class ProcessExecutor:
    def __init__(self):
        self.redis_connection = env_prepare()

    def start_job(self, deployment_path: str):
        job_details = load_details(deployment_path)
        self._push_pending_job(job_details)

    def _push_pending_job(self, job_details: dict):
        job_name = job_details["name"]
        # Push job details to redis
        self.redis_connection.hset(
            ProcessRedisName.JOB_DETAILS,
            job_name,
            json.dumps(job_details)
        )

        # Push job name to pending_job_tickets
        self.redis_connection.lpush(
            ProcessRedisName.PENDING_JOB_TICKETS,
            job_name
        )
        logger.info(f"Sending {job_name} into pending job tickets.")

    def stop_job(self, job_name: str):
        if not self.redis_connection.hexists(ProcessRedisName.JOB_DETAILS, job_name):
            logger.error(f"No such job '{job_name}' in Redis.")
            return

        # push job_name into kill_job_tickets
        self.redis_connection.lpush(
            ProcessRedisName.KILLED_JOB_TICKETS,
            job_name
        )
        logger.info(f"Sending {job_name} into killed job tickets.")

    def delete_job(self, job_name: str):
        # Stop job for running and pending job.
        self.stop_job(job_name)

        # Rm job details in Redis
        self.redis_connection.hdel(ProcessRedisName.JOB_DETAILS, job_name)

        # Rm job's log folder
        job_folder = os.path.expanduser(f"{LocalPaths.MARO_PROCESS}/{job_name}")
        shutil.rmtree(job_folder, True)
        logger.info(f"Remove local temporary log folder {job_folder}.")

    def get_job_logs(self, job_name):
        source_path = os.path.expanduser(f"{LocalPaths.MARO_PROCESS}/{job_name}")
        if not os.path.exists(source_path):
            logger.error(f"Cannot find the logs of {job_name}.")

        destination = os.path.join(os.getcwd(), job_name)
        if os.path.exists(destination):
            shutil.rmtree(destination)
        shutil.copytree(source_path, destination)
        logger.info(f"Dump logs in path: {destination}.")

    def list_job(self):
        # Get all jobs
        jobs = self.redis_connection.hgetall(ProcessRedisName.JOB_DETAILS)
        for job_name, job_details in jobs.items():
            job_name = job_name.decode()
            job_details = json.loads(job_details)
            if self.redis_connection.hexists(ProcessRedisName.RUNNING_JOB, job_name):
                job_details["job_status"] = "running"
            else:
                pending_jobs = self.redis_connection.lrange(ProcessRedisName.PENDING_JOB_TICKETS, 0, -1)
                pending_jobs = [job_name.decode() for job_name in pending_jobs]
                job_details["job_status"] = "pending" if job_name in pending_jobs else "finish"

            logger.info(job_details)

    def start_schedule(self, deployment_path: str):
        schedule_detail = load_details(deployment_path)
        # push schedule details to Redis
        self.redis_connection.hset(
            ProcessRedisName.JOB_DETAILS,
            schedule_detail["name"],
            json.dumps(schedule_detail)
        )

        job_list = schedule_detail["job_names"]
        # switch schedule details into job details
        job_detail = copy.deepcopy(schedule_detail)
        del job_detail["job_names"]

        for job_name in job_list:
            job_detail["name"] = job_name
            self._push_pending_job(job_detail)

    def stop_schedule(self, schedule_name: str):
        if self.redis_connection.hexists(ProcessRedisName.JOB_DETAILS, schedule_name):
            schedule_details = json.loads(self.redis_connection.hget(ProcessRedisName.JOB_DETAILS, schedule_name))
        else:
            logger.error(f"Cannot find {schedule_name} in Redis. Please check schedule name.")
            return

        job_list = schedule_details["job_names"]

        for job_name in job_list:
            self.stop_job(job_name)

    def get_job_details(self):
        jobs = self.redis_connection.hgetall(ProcessRedisName.JOB_DETAILS)
        for job_name, job_details_str in jobs.items():
            jobs[job_name] = json.loads(job_details_str)

        return list(jobs.values())

    def get_job_queue(self):
        pending_job_queue = self.redis_connection.lrange(
            ProcessRedisName.PENDING_JOB_TICKETS,
            0, -1
        )
        killed_job_queue = self.redis_connection.lrange(
            ProcessRedisName.KILLED_JOB_TICKETS,
            0, -1
        )
        return {
            "pending_jobs": pending_job_queue, 
            "killed_jobs": killed_job_queue
        }

    def get_static_resource(self):
        name_to_local_resource = self.redis_connection.hgetall("name_to_local_details")
        for node_name, node_details_str in name_to_local_resource.items():
            node_details = json.loads(node_details_str)
            name_to_local_resource[node_name] = node_details

        return name_to_local_resource

    def get_resource_usage(self, previous_length: int):
        usage_dict = {}
        usage_dict["cpu"] = self.redis_connection.lrange(
            f"process:cpu_usage_per_core",
            previous_length, -1
        )
        usage_dict["memory"] = self.redis_connection.lrange(
            f"process:memory_usage",
            previous_length, -1
        )
        usage_dict["gpu"] = self.redis_connection.lrange(
            f"process:gpu_memory_usage",
            previous_length, -1
        )

        return {"process": usage_dict}
