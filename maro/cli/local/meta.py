# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from os.path import expanduser, join

from maro.utils.utils import LOCAL_MARO_ROOT


LOCAL_ROOT = expanduser("~/.maro/proc")
LOCAL_STATE_PATH = join(LOCAL_ROOT, "state")
DEFAULT_DOCKER_FILE_PATH = join(LOCAL_MARO_ROOT, "docker_files", "dev.df")
DEFAULT_DOCKER_IMAGE_NAME = "maro-local"
DEFAULT_DOCKER_NETWORK = "MAROLOCAL"
DEFAULT_REDIS_CONTAINER_NAME = "maro-local-redis"


class RedisHashKey:
    """Record Redis elements name, and only for maro process"""
    JOB_CONF = "job_conf"
    JOB_DETAILS = "job_details"


class JobStatus:
    PENDING = "pending"
    IMAGE_BUILDING = "image_building"
    RUNNING = "running"
    ERROR = "error"
    REMOVED = "removed"
    FINISHED = "finished"
