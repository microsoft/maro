# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import logging
import os


class GlobalParams:
    PARALLELS = 5
    LOG_LEVEL = logging.INFO

    DEFAULT_REDIS_PORT = 6379
    DEFAULT_FLUENTD_PORT = 24224
    DEFAULT_SSH_PORT = 22


class GlobalPaths:
    MARO_CLUSTERS = "~/.maro/clusters"
    MARO_DATA = "~/.maro/data"
    MARO_TEST = "~/.maro/test"
    ABS_MARO_CLUSTERS = os.path.expanduser(MARO_CLUSTERS)
    ABS_MARO_DATA = os.path.expanduser(MARO_DATA)
    ABS_MARO_TEST = os.path.expanduser(MARO_TEST)

    MARO_LOCAL = "~/.maro-local"
    MARO_LOCAL_CLUSTER = "~/.maro-local/cluster"
    MARO_LOCAL_TMP = "~/.maro-local/tmp"
    ABS_MARO_LOCAL_CLUSTER = os.path.expanduser(MARO_LOCAL_CLUSTER)
    ABS_MARO_LOCAL_TMP = os.path.expanduser(MARO_LOCAL_TMP)

    MARO_SHARED = "~/.maro-shared"


class LocalParams:
    RESOURCE_REDIS_PORT = 7376
    RESOURCE_INFO = "local_resource:information"
    CPU_USAGE = "local_resource:cpu_usage_per_core"
    MEMORY_USAGE = "local_resource:memory_usage"
    GPU_USAGE = "local_resource:gpu_memory_usage"
