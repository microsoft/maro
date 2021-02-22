# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os


class NodeStatus:
    PENDING = "Pending"
    RUNNING = "Running"
    STOPPED = "Stopped"


class ContainerStatus:
    RUNNING = "running"
    EXITED = "exited"


class UserRole:
    ADMIN = "admin"


class JobStatus:
    PENDING = "pending"
    RUNNING = "running"
    KILLED = "killed"
    FINISH = "finish"
    FAILED = "failed"


class GrassParams:
    DEFAULT_API_SERVER_PORT = 51812


class GrassPaths:
    MARO_GRASS_LIB = "~/.maro/lib/grass"
    ABS_MARO_GRASS_LIB = os.path.expanduser(MARO_GRASS_LIB)
