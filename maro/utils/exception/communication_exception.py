# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .base_exception import MAROException


class RedisConnectionError(MAROException):
    """Failure to connect to redis, occurs in the proxy."""
    def __init__(self, msg: str = None):
        super().__init__(1001, msg)


class PeersMissError(MAROException):
    """Proxy do not have any expected peers."""
    def __init__(self, msg: str = None):
        super().__init__(1002, msg)


class InformationUncompletedError(MAROException):
    """No enough information from the Redis."""
    def __init__(self, msg: str = None):
        super().__init__(1003, msg)


class PeersConnectionError(MAROException):
    """Peers connection error, occurs in the driver."""
    def __init__(self, msg: str = None):
        super().__init__(1004, msg)


class DriverSendError(MAROException):
    """Failure to send message in the driver."""
    def __init__(self, msg: str = None):
        super().__init__(1005, msg)


class DriverReceiveError(MAROException):
    """Failure to receive message in the driver."""
    def __init__(self, msg: str = None):
        super().__init__(1006, msg)


class MessageSessionTypeError(MAROException):
    """The unrecognized session type, occurs in the ``SessionMessage``."""
    def __init__(self, msg: str = None):
        super().__init__(1007, msg)


class ConditionalEventSyntaxError(MAROException):
    """The syntax error of a conditional event."""
    def __init__(self, msg: str = None):
        super().__init__(1008, msg)


class DriverTypeError(MAROException):
    """The unrecognized driver type, occurs in the proxy."""
    def __init__(self, msg: str = None):
        super().__init__(1009, msg)


class SocketTypeError(MAROException):
    """The unrecognized socket type, occurs in the driver."""
    def __init__(self, msg: str = None):
        super().__init__(1010, msg)


class PeersDisconnectionError(MAROException):
    """Peers disconnection error, occurs in the driver. """
    def __init__(self, msg: str = None):
        super().__init__(1011, msg)


class PendingToSend(MAROException):
    """Temporary failure to send message, try to rejoin."""
    def __init__(self, msg: str = None):
        super().__init__(1012, msg)


class PeersRejoinTimeout(MAROException):
    """Failure to get enough peers during the max waiting time."""
    def __init__(self, msg: str = None):
        super().__init__(1013, msg)


__all__ = [
    "RedisConnectionError", "PeersMissError", "InformationUncompletedError", "DriverTypeError",
    "PeersConnectionError", "DriverSendError", "DriverReceiveError", "MessageSessionTypeError",
    "ConditionalEventSyntaxError", "SocketTypeError", "PeersDisconnectionError", "PendingToSend",
    "PeersRejoinTimeout"
]
