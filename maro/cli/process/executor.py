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
from maro.cli.process.utils.details import close_by_pid, get_redis_pid_by_port
from maro.cli.utils.abs_visible_executor import AbsVisibleExecutor
from maro.cli.utils.details_reader import DetailsReader
from maro.cli.utils.details_writer import DetailsWriter
from maro.cli.utils.params import GlobalPaths, LocalPaths, ProcessRedisName
from maro.cli.utils.resource_executor import LocalResourceExecutor
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)


class ProcessExecutor(AbsVisibleExecutor):
    def __init__(self, details: dict = None):
        self.details = details if details else \
            DetailsReader.load_cluster_details("process")

        # Connection with Redis
        redis_port = self.details["redis_info"]["port"]
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

    def create(self):
        logger.info("Starting MARO Multi-Process Mode.")
        if os.path.isdir(f"{GlobalPaths.ABS_MARO_CLUSTERS}/process"):
            logger.warning("Process mode has been created.")

        # Get environment setting
        DetailsWriter.save_cluster_details(
            cluster_name="process",
            cluster_details=self.details
        )

        # Start agents
        command = f"python {LocalPaths.MARO_PROCESS_AGENT}"
        _ = subprocess.Popen(command, shell=True)
        self._redis_connection.hset(ProcessRedisName.SETTING, "agent_status", 1)

        # Add connection to resource Redis.
        self._resource_redis.add_cluster()

        logger.info(f"MARO process mode setting: {self.details}")

    def delete(self):
        process_setting = self._redis_connection.hgetall(ProcessRedisName.SETTING)
        process_setting = {
            key.decode(): json.loads(value) for key, value in process_setting.items()
        }

        # Stop running jobs
        jobs = self._redis_connection.hgetall(ProcessRedisName.JOB_DETAILS)
        if jobs:
            for job_name, job_detail in jobs.items():
                job_detail = json.loads(job_detail)
                if job_detail["status"] == JobStatus.RUNNING:
                    close_by_pid(pid=job_detail["pid_list"], recursive=False)
                    logger.info(f"Stop running job {job_name.decode()}.")

        # Stop agents
        agent_status = int(process_setting["agent_status"])
        if agent_status:
            agent_pid = int(process_setting["agent_pid"])
            close_by_pid(pid=agent_pid, recursive=True)
            logger.info("Close agents.")
        else:
            logger.info("Agents is already closed.")

        # Stop Redis or clear Redis
        redis_mode = self.details["redis_mode"]
        if redis_mode == "MARO":
            redis_pid = get_redis_pid_by_port(self.details["redis_info"]["port"])
            close_by_pid(pid=redis_pid, recursive=False)
        else:
            self._redis_clear()

        # Rm connection from resource redis.
        self._resource_redis.sub_cluster()

        logger.info("Redis cleared.")

        # Remove local process file.
        shutil.rmtree(f"{GlobalPaths.ABS_MARO_CLUSTERS}/process", True)
        logger.info("Process mode has been deleted.")

    def _redis_clear(self):
        redis_keys = self._redis_connection.keys("process:*")
        for key in redis_keys:
            self._redis_connection.delete(key)

    def start_job(self, deployment_path: str):
        # Load start_job_deployment
        with open(deployment_path, "r") as fr:
            start_job_deployment = yaml.safe_load(fr)

        job_name = start_job_deployment["name"]
        start_job_deployment["status"] = JobStatus.PENDING
        # Push job details to redis
        self._redis_connection.hset(
            ProcessRedisName.JOB_DETAILS,
            job_name,
            json.dumps(start_job_deployment)
        )

        self._push_pending_job(job_name)

    def _push_pending_job(self, job_name: str):
        # Push job name to pending_job_tickets
        self._redis_connection.lpush(
            ProcessRedisName.PENDING_JOB_TICKETS,
            job_name
        )
        logger.info(f"Sending {job_name} into pending job tickets.")

    def stop_job(self, job_name: str):
        if not self._redis_connection.hexists(ProcessRedisName.JOB_DETAILS, job_name):
            logger.error(f"No such job '{job_name}' in Redis.")
            return

        # push job_name into kill_job_tickets
        self._redis_connection.lpush(
            ProcessRedisName.KILLED_JOB_TICKETS,
            job_name
        )
        logger.info(f"Sending {job_name} into killed job tickets.")

    def delete_job(self, job_name: str):
        # Stop job for running and pending job.
        self.stop_job(job_name)

        # Rm job details in Redis
        self._redis_connection.hdel(ProcessRedisName.JOB_DETAILS, job_name)

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
        jobs = self._redis_connection.hgetall(ProcessRedisName.JOB_DETAILS)
        for job_name, job_detail in jobs.items():
            job_name = job_name.decode()
            job_detail = json.loads(job_detail)

            logger.info(job_detail)

    def start_schedule(self, deployment_path: str):
        with open(deployment_path, "r") as fr:
            schedule_detail = yaml.safe_load(fr)

        # push schedule details to Redis
        self._redis_connection.hset(
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

            # Push job details to redis
            self._redis_connection.hset(
                ProcessRedisName.JOB_DETAILS,
                job_name,
                json.dumps(job_detail)
            )

            self._push_pending_job(job_name)

    def stop_schedule(self, schedule_name: str):
        if self._redis_connection.hexists(ProcessRedisName.JOB_DETAILS, schedule_name):
            schedule_details = json.loads(self._redis_connection.hget(ProcessRedisName.JOB_DETAILS, schedule_name))
        else:
            logger.error(f"Cannot find {schedule_name} in Redis. Please check schedule name.")
            return

        if "job_names" not in schedule_details.keys():
            logger.error(f"'{schedule_name}' is not a schedule.")
            return

        job_list = schedule_details["job_names"]

        for job_name in job_list:
            self.stop_job(job_name)

    def get_job_details(self):
        jobs = self._redis_connection.hgetall(ProcessRedisName.JOB_DETAILS)
        for job_name, job_details_str in jobs.items():
            jobs[job_name] = json.loads(job_details_str)

        return list(jobs.values())

    def get_job_queue(self):
        pending_job_queue = self._redis_connection.lrange(
            ProcessRedisName.PENDING_JOB_TICKETS,
            0, -1
        )
        killed_job_queue = self._redis_connection.lrange(
            ProcessRedisName.KILLED_JOB_TICKETS,
            0, -1
        )
        return {
            "pending_jobs": pending_job_queue,
            "killed_jobs": killed_job_queue
        }

    def get_resource(self):
        return self._resource_redis.get_local_resource()

    def get_resource_usage(self, previous_length: int):
        return self._resource_redis.get_local_resource_usage(previous_length)
