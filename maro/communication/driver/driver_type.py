# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum


class DriverType(Enum):
    ZMQ = "zmq_driver"  # The communication driver mode based on ZMQ
