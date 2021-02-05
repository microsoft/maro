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
    """Exception when access attribute that slot number greater than 1.

    This exception is caused when using invalid slice interface to access slots.
    """

    def __init__(self):
        super().__init__(2102, ERROR_CODE[2102])


class BackendsAppendToNonListAttributeException(MAROException):
    """Exception when append value to a non list attribute.
    """

    def __init__(self):
        super().__init__(2103, ERROR_CODE[2103])


class BackendsResizeNonListAttributeException(MAROException):
    """Exception when try to resize a non list attribute.
    """

    def __init__(self):
        super().__init__(2104, ERROR_CODE[2104])


class BackendsClearNonListAttributeException(MAROException):
    """Exception when try to clear a non list attribute.
    """

    def __init__(self):
        super().__init__(2105, ERROR_CODE[2105])


class BackendsInsertNonListAttributeException(MAROException):
    """Exception when try to insert a value to non list attribute.
    """

    def __init__(self):
        super().__init__(2106, ERROR_CODE[2106])


class BackendsRemoveFromNonListAttributeException(MAROException):
    """Exception when try to from a value to non list attribute.
    """

    def __init__(self):
        super().__init__(2107, ERROR_CODE[2107])


class BackendsAccessDeletedNodeException(MAROException):
    """Exception when try to access a deleted node.
    """

    def __init__(self):
        super().__init__(2108, ERROR_CODE[2108])


class BackendsInvalidNodeException(MAROException):
    """Exception when try to access a not exist node type.
    """

    def __init__(self):
        super().__init__(2109, ERROR_CODE[2109])


class BackendsInvalidAttributeException(MAROException):
    """Exception when try to access a not exist attribute type.
    """

    def __init__(self):
        super().__init__(2110, ERROR_CODE[2110])
