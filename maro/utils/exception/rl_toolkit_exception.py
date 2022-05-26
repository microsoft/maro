# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .base_exception import MAROException


class MissingTrainer(MAROException):
    """
    Raised when the trainer specified in the prefix of a policy name is missing.
    """

    def __init__(self, msg: str = None):
        super().__init__(4000, msg)
