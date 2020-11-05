# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .base_exception import MAROException


class UnsupportedAgentModeError(MAROException):
    """Unsupported agent mode error."""
    def __init__(self, msg: str = None):
        super().__init__(4001, msg)


class MissingShaperError(MAROException):
    """Missing shaper error."""
    def __init__(self, msg: str = None):
        super().__init__(4002, msg)


class WrongAgentModeError(MAROException):
    """Wrong agent mode error."""
    def __init__(self, msg: str = None):
        super().__init__(4003, msg)
