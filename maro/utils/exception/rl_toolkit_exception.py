# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .base_exception import MAROException


class InvalidExperience(MAROException):
    """
    Raised when the states, actions, rewards and next states passed to an ``ExperienceSet`` do not
    have the same length.
    """
    def __init__(self, msg: str = None):
        super().__init__(4000, msg)


class MissingOptimizer(MAROException):
    """Raised when the optimizers are missing when calling CoreModel's step() method."""
    def __init__(self, msg: str = None):
        super().__init__(4001, msg)
