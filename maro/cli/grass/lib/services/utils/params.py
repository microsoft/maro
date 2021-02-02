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


class JobStatus:
    PENDING = "pending"
    RUNNING = "running"
    KILLED = "killed"
    FINISH = "finish"
    FAILED = "failed"


class UserRole:
    ADMIN = "admin"


class Paths:
    MARO_SHARED = "~/.maro-shared"
    ABS_MARO_SHARED = os.path.expanduser(path=MARO_SHARED)

    MARO_LOCAL = "~/.maro-local"
    ABS_MARO_LOCAL = os.path.expanduser(path=MARO_LOCAL)
