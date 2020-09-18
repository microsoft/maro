
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from .base_exception import MAROException
from .error_code import ERROR_CODE


class MetaTimestampNotExist(MAROException):
    def __init__(self):
        super().__init__(2000, ERROR_CODE[2000])


class EcrGeneratorInvalidParkingDuration(MAROException):
    def __init__(self):
        super().__init__(2001, ERROR_CODE[2001])