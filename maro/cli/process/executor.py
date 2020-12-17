# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import json
import os
import shutil

from maro.cli.process.utils.details import env_preset, load_details
from maro.cli.utils.params import LocalPaths, ProcessRedisName
from maro.utils.exception.cli_exception import CliException
from maro.utils.logger import CliLogger

logger = CliLogger(name=__name__)


class ProcessExecutor:
    def __init__(self):
        self.redis_connection = env_preset()

    def start_job(self, deployment_path: str):
        job_details = load_details(deployment_path)
        self._push_pending_job(job_details)

    def stop_job(self, job_name: str):
        # push job_name into kill_job_tickets
        self.redis_connection.lpush(
            ProcessRedisName.KILLED_JOB_TICKETS,
            job_name
        )

    def delete_job(self, job_name: str):
        # Stop job for running and pending job.
        self.stop_job(job_name)

        # rm Redis
        self.redis_connection.hdel(ProcessRedisName.JOB_DETAILS, job_name)

        # rm dir
        job_folder = os.path.expanduser(f"{LocalPaths.MARO_PROCESS}/{job_name}")
        shutil.rmtree(job_folder, True)

    def get_job_logs(self, job_name):
        source_path = os.path.expanduser(f"{LocalPaths.MARO_PROCESS}/{job_name}")
        if not os.path.exists(source_path):
            logger.error(f"Cannot find logs about {job_name}.")

        destination = shutil.copytree(source_path, os.path.join(os.getcwd(), job_name))
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
                if job_name in pending_jobs:
                    job_details["job_status"] = "pending"
                else:
                    job_details["job_status"] = "finish"
            logger.info(job_details)

    def _push_pending_job(self, job_details: dict):
        job_name = job_details["name"]
        # Push job details to redis
        self.redis_connection.hset(
            ProcessRedisName.JOB_DETAILS,
            job_name,
            json.dumps(job_details)
        )

        # Push job to pending_job_tickets
        self.redis_connection.lpush(
            ProcessRedisName.PENDING_JOB_TICKETS,
            job_name
        )

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
        job_detail["schedule_name"] = job_detail["name"]
        del job_detail["job_names"]

        for job_name in job_list:
            job_detail["name"] = job_name
            self._push_pending_job(job_detail)

    def stop_schedule(self, schedule_name: str):
        if self.redis_connection.hexists(ProcessRedisName.JOB_DETAILS, schedule_name):
            schedule_details = json.loads(self.redis_connection.hget(ProcessRedisName.JOB_DETAILS, schedule_name))
        else:
            raise CliException(f"Cannot find {schedule_name} in Redis. Please check schedule name.")
        job_list = schedule_details["job_names"]

        for job_name in job_list:
            self.redis_connection.lpush(
                ProcessRedisName.KILLED_JOB_TICKETS,
                job_name
            )
