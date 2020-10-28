# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from maro.utils.exception import MAROException


class UnsupportedAgentManagerModeError(MAROException):
    """Unsupported agent mode."""
    def __init__(self, msg: str = None):
        super().__init__(4001, msg)


class MissingShaperError(MAROException):
    """Missing shaper."""
    def __init__(self, msg: str = None):
        super().__init__(4002, msg)


class WrongAgentManagerModeError(MAROException):
    """Wrong agent manager mode."""
    def __init__(self, msg: str = None):
        super().__init__(4003, msg)


class StoreMisalignmentError(MAROException):
    """Raised when a ``put`` operation on a ``ColumnBasedStore`` would cause the underlying lists to have different
    sizes."""
    def __init__(self, msg: str = None):
        super().__init__(4004, msg)
