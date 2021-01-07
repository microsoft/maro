# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


class NodeStatus:
    PENDING = "Pending"
    RUNNING = "Running"
    STOPPED = "Stopped"


class ContainerStatus:
    RUNNING = "running"
    EXITED = "exited"


class GrassParams:
    DEFAULT_API_SERVER_PORT = 51812
