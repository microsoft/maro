# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import json
import os
import shutil

from maro.cli.process.utils.details import env_preset, load_details
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
            "local_process:killed_job_tickets",
            job_name
        )

    def delete_job(self, job_name: str):
        # check status
        status = self.redis_connection.hget("local_process:job_status", job_name).decode()

        if status == "runtime":
            self.stop_job(job_name)

        # rm Redis
        self.redis_connection.hdel("local_process:job_details", job_name)
        self.redis_connection.hdel("local_process:job_status", job_name)

        # rm dir
        job_folder = os.path.expanduser(f"~/.maro/local/{job_name}")
        shutil.rmtree(job_folder, True)

    def get_job_logs(self, job_name):
        source_path = os.path.expanduser(f"~/.maro/local/{job_name}")
        if not os.path.exists(source_path):
            logger.error(f"Cannot find logs about {job_name}.")

        destination = shutil.copytree(source_path, os.path.join(os.getcwd(), job_name))
        logger.info(f"Dump logs in path: {destination}.")

    def list_job(self):
        # Get all jobs
        jobs = self.redis_connection.hgetall("local_process:job_status")
        jobs = {job.decode(): job_status.decode() for job, job_status in jobs.items()}
        logger.info(jobs)

    def _push_pending_job(self, job_details: dict):
        job_name = job_details["name"]
        # Push job details to redis
        self.redis_connection.hset(
            "local_process:job_details",
            job_name,
            json.dumps(job_details)
        )

        # Set job status in redis
        self.redis_connection.hset(
            "local_process:job_status",
            job_name,
            json.dumps("pending")
        )

        # Push job to pending_job_tickets
        self.redis_connection.lpush(
            "local_process:pending_job_tickets",
            job_name
        )

    def start_schedule(self, deployment_path: str):
        schedule_detail = load_details(deployment_path)
        # push schedule details to Redis
        self.redis_connection.hset(
            "local_process:job_details",
            schedule_detail["name"],
            json.dumps(schedule_detail)
        )

        job_list = schedule_detail["job_names"]
        # switch schedule details into job details
        job_detail = copy.deepcopy(schedule_detail)
        job_detail["parallel"] = 1
        job_detail["schedule_name"] = job_detail["name"]
        del job_detail["job_names"]

        for job_name in job_list:
            job_detail["name"] = job_name
            self._push_pending_job(job_detail)

    def stop_schedule(self, schedule_name: str):
        schedule_details = json.loads(self.redis_connection.hget("local_process:job_details", schedule_name))
        job_list = schedule_details["job_names"]

        for job_name in job_list:
            self.redis_connection.lpush(
                "local_process:killed_job_tickets",
                job_name
            )
