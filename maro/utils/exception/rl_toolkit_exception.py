# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .base_exception import MAROException


class AgentManagerModeError(MAROException):
    """Wrong agent manager mode."""
    def __init__(self, msg: str = None):
        super().__init__(4000, msg)


class MissingShaper(MAROException):
    """Missing shaper."""
    def __init__(self, msg: str = None):
        super().__init__(4001, msg)


class StoreMisalignment(MAROException):
    """Raised when a ``put`` operation on a ``SimpleStore`` would cause the underlying lists to have different
    sizes."""
    def __init__(self, msg: str = None):
        super().__init__(4002, msg)


class InvalidEpisode(MAROException):
    """Raised when the ``max_episode`` passed to the the ``SimpleLearner``'s ``train`` method is negative and not -1."""
    def __init__(self, msg: str = None):
        super().__init__(4003, msg)


class InfiniteTrainingLoop(MAROException):
    """Raised when the ``SimpleLearner``'s training loop becomes infinite."""
    def __init__(self, msg: str = None):
        super().__init__(4004, msg)


class MissingOptimizer(MAROException):
    """Raised when the optimizers are missing when calling LearningModel's step() method."""
    def __init__(self, msg: str = None):
        super().__init__(4005, msg)


class UnrecognizedTask(MAROException):
    """Raised when a LearningModel has task names that are not unrecognized by an algorithm."""
    def __init__(self, msg: str = None):
        super().__init__(4006, msg)


class NNStackDimensionError(MAROException):
    """Raised when a learning module's input dimension is incorrect."""
    def __init__(self, msg: str = None):
        super().__init__(4007, msg)
