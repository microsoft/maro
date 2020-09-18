# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from .base_exception import MAROException
from .error_code import ERROR_CODE


class BackendsGetItemInvalidException(MAROException):
    def __init__(self):
        super().__init__(2100, ERROR_CODE[2100])


class BackendsSetItemInvalidException(MAROException):
    def __init__(self):
        super().__init__(2101, ERROR_CODE[2101])

class BackendsArrayAttributeAccessException(MAROException):
    def __init__(self):
        super().__init__(2102, ERROR_CODE[2102])