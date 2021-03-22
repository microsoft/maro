# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .base_exception import MAROException


class StoreMisalignment(MAROException):
    """Raised when a ``put`` operation on a ``SimpleStore`` would cause the underlying lists to have different
    sizes."""
    def __init__(self, msg: str = None):
        super().__init__(4000, msg)


class MissingOptimizer(MAROException):
    """Raised when the optimizers are missing when calling CoreModel's step() method."""
    def __init__(self, msg: str = None):
        super().__init__(4001, msg)


class UnrecognizedTask(MAROException):
    """Raised when a CoreModel has task names that are not unrecognized by an algorithm."""
    def __init__(self, msg: str = None):
        super().__init__(4002, msg)
