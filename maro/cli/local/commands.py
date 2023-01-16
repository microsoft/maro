# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import shutil
import subprocess
import sys
import time
from os import makedirs
from os.path import abspath, dirname, exists, expanduser, join

import redis
import yaml

from maro.cli.utils.common import close_by_pid, show_log
from maro.rl.workflows.config import ConfigParser
from maro.utils.logger import CliLogger
from maro.utils.utils import LOCAL_MARO_ROOT

from .utils import (
    JobStatus,
    RedisHashKey,
    start_redis,
    start_rl_job,
    start_rl_job_with_docker_compose,
    stop_redis,
    stop_rl_job_with_docker_compose,
)

# metadata
LOCAL_ROOT = expanduser("~/.maro/local")
SESSION_STATE_PATH = join(LOCAL_ROOT, "session.json")
DOCKERFILE_PATH = join(LOCAL_MARO_ROOT, "docker_files", "dev.df")
DOCKER_IMAGE_NAME = "maro-local"
DOCKER_NETWORK = "MAROLOCAL"

# display
NO_JOB_MANAGER_MSG = """No job manager found. Run "maro local init" to start the job manager first."""
NO_JOB_MSG = """No job named {} found. Run "maro local job ls" to view existing jobs."""
JOB_LS_TEMPLATE = "{JOB:12}{STATUS:15}{STARTED:20}"

logger = CliLogger(name="MARO-LOCAL")


# helper functions
def get_redis_conn(port=None):
    if port is None:
        try:
            with open(SESSION_STATE_PATH, "r") as fp:
                port = json.load(fp)["port"]
        except FileNotFoundError:
            logger.error(NO_JOB_MANAGER_MSG)
            return

    try:
        redis_conn = redis.Redis(host="localhost", port=port)
        redis_conn.ping()
        return redis_conn
    except redis.exceptions.ConnectionError:
        logger.error(NO_JOB_MANAGER_MSG)


# Functions executed on CLI commands
def run(conf_path: str, containerize: bool = False, evaluate_only: bool = False, **kwargs):
    # Load job configuration file
    parser = ConfigParser(conf_path)
    if containerize:
        try:
            start_rl_job_with_docker_compose(
                parser,
                LOCAL_MARO_ROOT,
                DOCKERFILE_PATH,
                DOCKER_IMAGE_NAME,
                evaluate_only=evaluate_only,
            )
        except KeyboardInterrupt:
            stop_rl_job_with_docker_compose(parser.config["job"], LOCAL_MARO_ROOT)
    else:
        try:
            start_rl_job(parser, LOCAL_MARO_ROOT, evaluate_only=evaluate_only)
        except KeyboardInterrupt:
            sys.exit(1)


def init(
    port: int = 19999,
    max_running: int = 3,
    query_every: int = 5,
    timeout: int = 3,
    containerize: bool = False,
    **kwargs,
):
    if exists(SESSION_STATE_PATH):
        with open(SESSION_STATE_PATH, "r") as fp:
            session_state = json.load(fp)
        logger.warning(
            f"Local job manager is already running at port {session_state['port']}. "
            f"Run 'maro local job add/rm' to add / remove jobs.",
        )
        return

    start_redis(port)

    # Start job manager
    command = ["python", join(dirname(abspath(__file__)), "job_manager.py")]
    job_manager = subprocess.Popen(
        command,
        env={
            "PYTHONPATH": LOCAL_MARO_ROOT,
            "MAX_RUNNING": str(max_running),
            "QUERY_EVERY": str(query_every),
            "SIGTERM_TIMEOUT": str(timeout),
            "CONTAINERIZE": str(containerize),
            "REDIS_PORT": str(port),
            "LOCAL_MARO_ROOT": LOCAL_MARO_ROOT,
            "DOCKER_IMAGE_NAME": DOCKER_IMAGE_NAME,
            "DOCKERFILE_PATH": DOCKERFILE_PATH,
        },
    )

    # Dump environment setting
    makedirs(LOCAL_ROOT, exist_ok=True)
    with open(SESSION_STATE_PATH, "w") as fp:
        json.dump({"port": port, "job_manager_pid": job_manager.pid, "containerized": containerize}, fp)

    # Create log folder
    logger.info("Local job manager started")


def exit(**kwargs):
    try:
        with open(SESSION_STATE_PATH, "r") as fp:
            session_state = json.load(fp)
    except FileNotFoundError:
        logger.error(NO_JOB_MANAGER_MSG)
        return

    redis_conn = get_redis_conn()

    # Mark all jobs as REMOVED and let the job manager terminate them properly.
    job_details = redis_conn.hgetall(RedisHashKey.JOB_DETAILS)
    if job_details:
        for job_name, details in job_details.items():
            details = json.loads(details)
            details["status"] = JobStatus.REMOVED
            redis_conn.hset(RedisHashKey.JOB_DETAILS, job_name, json.dumps(details))
            logger.info(f"Gracefully terminating job {job_name.decode()}")

    # Stop job manager
    close_by_pid(int(session_state["job_manager_pid"]))

    # Stop Redis
    stop_redis(session_state["port"])

    # Remove dump folder.
    shutil.rmtree(LOCAL_ROOT, True)

    logger.info("Local job manager terminated.")


def add_job(conf_path: str, **kwargs):
    redis_conn = get_redis_conn()
    if not redis_conn:
        return

    # Load job configuration file
    with open(conf_path, "r") as fr:
        conf = yaml.safe_load(fr)

    job_name = conf["job"]
    if redis_conn.hexists(RedisHashKey.JOB_DETAILS, job_name):
        logger.error(f"A job named '{job_name}' has already been added.")
        return

    # Push job config to redis
    redis_conn.hset(RedisHashKey.JOB_CONF, job_name, json.dumps(conf))
    details = {
        "status": JobStatus.PENDING,
        "added": time.time(),
    }
    redis_conn.hset(RedisHashKey.JOB_DETAILS, job_name, json.dumps(details))


def remove_jobs(job_names, **kwargs):
    redis_conn = get_redis_conn()
    if not redis_conn:
        return

    for job_name in job_names:
        details = redis_conn.hget(RedisHashKey.JOB_DETAILS, job_name)
        if not details:
            logger.error(f"No job named '{job_name}' has been scheduled or started.")
        else:
            details = json.loads(details)
            details["status"] = JobStatus.REMOVED
            redis_conn.hset(RedisHashKey.JOB_DETAILS, job_name, json.dumps(details))
            logger.info(f"Removed job {job_name}")


def describe_job(job_name, **kwargs):
    redis_conn = get_redis_conn()
    if not redis_conn:
        return

    details = redis_conn.hget(RedisHashKey.JOB_DETAILS, job_name)
    if not details:
        logger.error(NO_JOB_MSG.format(job_name))
        return

    details = json.loads(details)
    err = "error_message" in details
    if err:
        err_msg = details["error_message"].split("\n")
        del details["error_message"]

    logger.info(details)
    if err:
        for line in err_msg:
            logger.info(line)


def get_job_logs(job_name: str, tail: int = -1, **kwargs):
    redis_conn = get_redis_conn()
    if not redis_conn.hexists(RedisHashKey.JOB_CONF, job_name):
        logger.error(NO_JOB_MSG.format(job_name))
        return

    conf = json.loads(redis_conn.hget(RedisHashKey.JOB_CONF, job_name))
    show_log(conf["log_path"], tail=tail)


def list_jobs(**kwargs):
    redis_conn = get_redis_conn()
    if not redis_conn:
        return

    def get_time_diff_string(time_diff):
        time_diff = int(time_diff)
        days = time_diff // (3600 * 24)
        if days:
            return f"{days} days"

        hours = time_diff // 3600
        if hours:
            return f"{hours} hours"

        minutes = time_diff // 60
        if minutes:
            return f"{minutes} minutes"

        return f"{time_diff} seconds"

    # Header
    logger.info(JOB_LS_TEMPLATE.format(JOB="JOB", STATUS="STATUS", STARTED="STARTED"))
    for job_name, details in redis_conn.hgetall(RedisHashKey.JOB_DETAILS).items():
        job_name = job_name.decode()
        details = json.loads(details)
        if "start_time" in details:
            time_diff = f"{get_time_diff_string(time.time() - details['start_time'])} ago"
            logger.info(JOB_LS_TEMPLATE.format(JOB=job_name, STATUS=details["status"], STARTED=time_diff))
        else:
            logger.info(JOB_LS_TEMPLATE.format(JOB=job_name, STATUS=details["status"], STARTED=JobStatus.PENDING))
