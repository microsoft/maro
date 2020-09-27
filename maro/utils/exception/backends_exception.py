# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from .base_exception import MAROException
from .error_code import ERROR_CODE


class BackendsGetItemInvalidException(MAROException):
    """Exception if the parameters is invalid when getting item from backend.


    Usually this exception is caused by invalid node or attribute index.
    """

    def __init__(self):
        super().__init__(2100, ERROR_CODE[2100])


class BackendsSetItemInvalidException(MAROException):
    """Exception if the parameter is invalid when setting item from backend.


    Usually this exception is caused by invalid node or attribute index.
    """

    def __init__(self):
        super().__init__(2101, ERROR_CODE[2101])


class BackendsArrayAttributeAccessException(MAROException):
    """Exception then access attribute that slot number greater than 1.

    This exception is caused when using invalid slice interface to access slots.
    """

    def __init__(self):
        super().__init__(2102, ERROR_CODE[2102])
