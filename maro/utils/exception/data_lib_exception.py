
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from .base_exception import MAROException
from .error_code import ERROR_CODE


class MetaTimestampNotExist(MAROException):
    """Exception that there is no timestamp specified in meta file, as this is required field."""

    def __init__(self):
        super().__init__(2000, ERROR_CODE[2000])


class CimGeneratorInvalidParkingDuration(MAROException):
    """Exception when parking duration is less than 0 in CIM topology file."""

    def __init__(self):
        super().__init__(2001, ERROR_CODE[2001])
