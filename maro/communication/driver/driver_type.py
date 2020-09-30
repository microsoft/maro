# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum


class DriverType(Enum):
    """Communication driver categories.

    - ZMQ: The communication driver mode based on ``ZMQ``.
    """
    ZMQ = "zmq_driver"
