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
    MARO_LIB = "~/.maro/lib"
    MARO_GRASS_LIB = "~/.maro/lib/grass"
    MARO_K8S_LIB = "~/.maro/lib/k8s"
    MARO_CLUSTERS = "~/.maro/clusters"
    MARO_DATA = "~/.maro/data"
    MARO_TEST = "~/.maro/test"
    MARO_LOCAL_TMP = "~/.maro-local/tmp"

    ABS_MARO_LIB = os.path.expanduser(MARO_LIB)
    ABS_MARO_GRASS_LIB = os.path.expanduser(MARO_GRASS_LIB)
    ABS_MARO_K8S_LIB = os.path.expanduser(MARO_K8S_LIB)
    ABS_MARO_CLUSTERS = os.path.expanduser(MARO_CLUSTERS)
    ABS_MARO_DATA = os.path.expanduser(MARO_DATA)
    ABS_MARO_TEST = os.path.expanduser(MARO_TEST)
    ABS_MARO_LOCAL_TMP = os.path.expanduser(MARO_LOCAL_TMP)
